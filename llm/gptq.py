import math
import time

import torch
import torch.nn as nn
import transformers
import sys
sys.path.append("../")
from utils.quant import *
from opt import plot_weight_heatmap

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# FP4(E2M1) hardcoded LUT (positive values only)
# index = (exponent << 1) | mantissa
FP4_E2M1_LUT = torch.tensor(
    [0.0,   # E=00 M=0
     0.5,   # E=00 M=1
     1.0,   # E=01 M=0
     1.5,   # E=01 M=1
     2.0,   # E=10 M=0
     3.0,   # E=10 M=1
     4.0,   # E=11 M=0
     6.0]   # E=11 M=1
)

def fp4_e2m1_decompose(tensor):
    """
    Hardcoded FP4(E2M1) decomposition.
    Returns: sign, exponent_bits (0–3), mantissa_bit (0/1)
    """
    x = tensor.clone()

    # sign bit
    sign = (x < 0).int()
    x = x.abs()

    # flatten for vectorized LUT match
    x_flat = x.view(-1, 1)
    lut = FP4_E2M1_LUT.to(x.device).view(1, -1)

    # nearest FP4 value
    idx = torch.argmin((x_flat - lut).abs(), dim=1)

    exponent = (idx >> 1).view(x.shape)   # high bit
    mantissa = (idx & 1).view(x.shape)    # low bit

    return sign, exponent, mantissa

def plot_fp4_exponent_heatmap(Wq, title, filename):
    _, e, _ = fp4_e2m1_decompose(Wq)

    plot_weight_heatmap(
        e.float(),
        title + " (FP4 Exponent Bits)",
        filename
    )

def plot_fp4_mantissa_heatmap(Wq, title, filename):
    _, _, m = fp4_e2m1_decompose(Wq)

    plot_weight_heatmap(
        m.float(),
        title + " (FP4 Mantissa Bit)",
        filename
    )

def fp4_bits_to_str(sign, exp, man):
    return f"{sign}-{exp:02b}-{man}"

class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, prunen=0, prunem=0, plot=False, name=""
    ):
        # 打印N:M
        if prunen != 0:
            print(f"Applying {prunen}:{prunem} pruning during quantization.")
        # 打印blocksize
        print(f"Using blocksize of {blocksize} for quantization.")
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            # print("Conv2d")
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            # print("Conv1d")
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            # print("HINV Shape", W.shape, W1.shape, Hinv.shape, Hinv1.shape)
            mask_buffer = None
            mean_buffer = None
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                # print("w,d Shape", w.shape, d)
                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]
                
                if prunen != 0 and i % prunem == 0:
                    if i + prunem <= count:
                        w_group = W1[:, i:(i + prunem)].clone()
                    
                        # scores = w_group.abs()

                        diag_group = torch.tensor([Hinv1[j, j] for j in range(i, i + prunem)], device=self.dev)
                        scores = (w_group ** 2) / (diag_group ** 2)
                        
                        _, indices_to_prune = torch.topk(scores, k=prunen, dim=1, largest=False)
                        mask_buffer = torch.ones_like(w_group, dtype=torch.bool)
                        mask_buffer.scatter_(dim=1, index=indices_to_prune, value=False)

                        with torch.no_grad():
                            # case1: mean of kept weights
                            # kept_sum = (w_group * mask_buffer).sum(dim=1) 
                            # mean_buffer = kept_sum / (prunem - prunen) 

                            # case2: mean of original weights
                            # kept_sum = w_group.sum(dim=1) 
                            # mean_buffer = kept_sum / prunem

                            # case3: mean of pruned weights
                            pruned_sum = (w_group * (~mask_buffer)).sum(dim=1)
                            mean_buffer = pruned_sum / prunen

                            # if i1 == 0 and i == 0 and plot:
                            #     plot_weight_heatmap(w_group * (~mask_buffer), f"Pruned Weights at Block {i}", f"pruned_weights_{name}_block_0.png")
                            #     plot_weight_heatmap(mask_buffer, f"Pruning Mask at Block {i}", f"pruning_mask_{name}_block_0.png")

                            # case4: zero compensation
                            # mean_buffer = torch.zeros_like(w)
                    else:
                        mask_buffer = None
                        mean_buffer = None
                
                # If pruning mask exists, replace pruned entries in `w` with the
                # compensation mean before quantizing so quantization operates on
                # the pruned-compensated weight vector.
                if mask_buffer is not None:
                    col_mask = mask_buffer[:, i % prunem]
                    col_mean = mean_buffer
                    w_to_quant = torch.where(col_mask, w, col_mean)
                else:
                    w_to_quant = w

                q = quantize(w_to_quant.unsqueeze(1), self.quantizer.scale).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

                if i1 == 0 and i == prunem - 1 and plot and mask_buffer is not None:
                    # plot_weight_heatmap(Q1[:, :prunem] * (~mask_buffer), f"Quantized Pruned Weights at Block {i}", f"quantized_weights_{name}_block_0.png")
                    # plot_fp4_mantissa_heatmap((Q1[:, :prunem] * (~mask_buffer)) / self.quantizer.scale, f"FP4 Mantissa at Block {i}", f"fp4_mantissa_{name}_block_0.png")

                    fp4_vals = (Q1[:, :prunem] * (~mask_buffer)) / self.quantizer.scale

                    s, e, m = fp4_e2m1_decompose(fp4_vals)

                    print(f"\n[FP4 bits at block {i} | layer {name}]")

                    rows, cols = fp4_vals.shape
                    for r in range(32):
                        line = []
                        for c in range(32):
                            if not mask_buffer[r, c]:
                                # 被 prune 的位置
                                bitstr = f"{s[r,c].item()}-{e[r,c].item():02b}-{m[r,c].item()}"
                                line.append(bitstr)
                            else:
                                line.append("  .   ")
                        print(" ".join(line))
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
