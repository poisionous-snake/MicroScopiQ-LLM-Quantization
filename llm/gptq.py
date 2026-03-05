import math
import time

import torch
import torch.nn as nn
import transformers
import sys
sys.path.append("../")
from utils.quant import *


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
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, prunen=0, prunem=0, plot=False
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
        H_diag = torch.diag(H).clone()
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

                q = quantize(w.unsqueeze(1), self.quantizer.scale).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

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
        
        # ==================== 修改部分：N:M 结构化剪枝 ====================
        if prunen != 0 and prunem != 0:
            print(f"Applying post-quantization wanda {prunen}:{prunem} pruning.")
            out_features, in_features = Q.shape

            act_norm = torch.sqrt(H_diag + 1e-8)  # shape: (in_features,)
            if actorder:
                act_norm = act_norm[invperm]

            W_metric = torch.abs(Q) * act_norm.view(1, -1)
            
            # 针对 N:M，通常在输入特征维度（in_features）进行分组
            if in_features % prunem == 0:
                # 1. 重塑形状为 (out_features, 组数, M)
                W_temp = Q.view(out_features, -1, prunem)
                M_temp = W_metric.view(out_features, -1, prunem)

                # 2. 找到每组中绝对值最大的前 N 个元素的索引
                # topk 会返回前 prunen 个最大值的索引
                _, topk_indices = torch.topk(M_temp, prunen, dim=2, largest=True)
                
                # 3. 创建掩码并应用
                mask = torch.zeros_like(W_temp, dtype=torch.bool)
                mask.scatter_(2, topk_indices, True)
                
                # # 4. 计算被剪掉部分的平均值
                # # 提取出非 Top-N 的元素，其余位置设为 0 以便求和
                # pruned_elements = torch.where(~mask, W_temp, torch.zeros_like(W_temp))

                # # 每组被剪掉元素的总和
                # pruned_sum = torch.sum(pruned_elements, dim=2, keepdim=True)

                # # 每组被剪掉元素的个数
                # num_pruned = prunem - prunen

                # # 计算均值
                # pruned_mean = pruned_sum / num_pruned
                
                # # 5. 均值填充：Top-N 位置保留原值，非 Top-N 位置替换为均值
                # W_final = torch.where(mask, W_temp, pruned_mean)

                # # 5. 0填充
                # W_final = torch.where(mask, W_temp, torch.zeros_like(W_temp))

                # 4 activation-aware replacement
                act2 = act_norm.view(1, -1, prunem)

                pruned = ~mask

                num = torch.sum(W_temp * act2 * pruned, dim=2, keepdim=True)
                den = torch.sum(act2 * pruned, dim=2, keepdim=True) + 1e-8 

                replacement = num / den

                # 5 fill
                W_final = torch.where(mask, W_temp, replacement)

                # --- 新增：FP4 比特打印逻辑 (调试用) ---
                if plot:
                # 提取被剪枝位置（即 mask 为 False 的位置）的值
                # 为了观察 FP4 比特，我们需要除以 scale 还原到量化空间
                    current_scale = self.quantizer.scale

                    if current_scale.dim() == 2:
                        # 扩展 scale 维度到 (out_features, 1, 1) 以匹配 (out_features, groups, prunem)
                        scale_reshaped = current_scale.unsqueeze(2)
                    else:
                        scale_reshaped = current_scale

                    fp4_query_vals = (W_final / scale_reshaped)
                    
                    # 获取比特分解
                    s, e, m = fp4_e2m1_decompose(fp4_query_vals)
                    
                    print(f"\n[FP4 Bits for Pruned Elements (Replaced by Mean) | {prunen}:{prunem}]")
                    # 打印前 32xM 范围内的结构
                    rows_to_print = min(32, out_features)
                    groups_to_print = W_final.shape[1]

                    for r in range(rows_to_print):
                        group_bits = []
                        for i in range(prunem):
                            is_topn = mask[r, 0, i]
                            if not is_topn:
                                # 被剪枝的位置，现在显示的是均值的 FP4 比特
                                bitstr = f"{s[r, 0, i].item()}-{e[r, 0, i].item():02b}-{m[r, 0, i].item()}"
                                group_bits.append(f"{bitstr}")
                            else:
                                # 保留的 Top-N 位置
                                group_bits.append("   .  ")
                        
                        # 每个 group 打印完后直接输出并换行
                        print(" ".join(group_bits))

                # 6. 还原回原始二维形状
                Q = W_final.view(out_features, in_features)
            else:
                print(f"Warning: in_features({in_features}) is not divisible by {prunem}. Skipping N:M.")
        # ================================================================

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
