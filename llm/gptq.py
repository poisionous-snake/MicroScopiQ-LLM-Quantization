import math
import time

import torch
import numpy as np
import torch.nn as nn
import transformers
from sklearn.cluster import KMeans
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

# FP8(E4M3) hardcoded LUT (positive values only)
# index = (exponent << 3) | mantissa
# Exponent bias = 7, range -7 to 8
FP8_E4M3_LUT = torch.tensor([
    # exp=0 (2^-7)
    0.0078125, 0.0087890625, 0.009765625, 0.0107421875, 0.01171875, 0.0126953125, 0.013671875, 0.0146484375,
    # exp=1 (2^-6)
    0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 0.02734375, 0.029296875,
    # exp=2 (2^-5)
    0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375,
    # exp=3 (2^-4)
    0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875,
    # exp=4 (2^-3)
    0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375,
    # exp=5 (2^-2)
    0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875,
    # exp=6 (2^-1)
    0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375,
    # exp=7 (2^0)
    1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875,
    # exp=8 (2^1)
    2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
    # exp=9 (2^2)
    4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
    # exp=10 (2^3)
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    # exp=11 (2^4)
    16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
    # exp=12 (2^5)
    32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0,
    # exp=13 (2^6)
    64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0,
    # exp=14 (2^7)
    128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0,
    # exp=15 (2^8)
    256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0,
])

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

def fp8_e4m3_decompose(tensor):
    """
    Hardcoded FP8(E4M3) decomposition.
    Returns: sign, exponent_bits (0–15), mantissa_bits (0–7)
    """
    x = tensor.clone()

    # sign bit
    sign = (x < 0).int()
    x = x.abs()

    # flatten for vectorized LUT match
    x_flat = x.view(-1, 1)
    lut = FP8_E4M3_LUT.to(x.device).view(1, -1)

    # nearest FP8 value
    idx = torch.argmin((x_flat - lut).abs(), dim=1)

    exponent = (idx >> 3).view(x.shape)   # high 4 bits
    mantissa = (idx & 7).view(x.shape)    # low 3 bits

    return sign, exponent, mantissa

def fp4_bits_to_str(sign, exp, man):
    return f"{sign}-{exp:02b}-{man}"

def kmeans_exp_vq(exp_g, man_g, k=16, iters=10):
    """
    exp_g: [N, d]
    man_g: [N, d]
    """
    N = exp_g.shape[0]

    # === 初始化 centroid（从数据中采样）===
    unique_exp = torch.unique(exp_g, dim=0) # [U, d]
    # print(f"Unique exponent patterns: {unique_exp.shape[0]}")
    idx = torch.randperm(unique_exp.shape[0])[:k]
    centroids = unique_exp[idx].float() # [k, d]

    for _ in range(iters):
        # # === SIMPLEST DIST ===
        # dist = ((exp_g.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(-1)  # [N, k]
        
        # # === 2^exp DIST ===
        # dist = ((2 ** (exp_g.unsqueeze(1) - 1)  - 2 ** (centroids.unsqueeze(0) - 1)) ** 2).sum(-1)  # [N, k]
        
        # === +man LUT DIST ===
        LUT = FP8_E4M3_LUT.to(exp_g.device) # [128] # TODO：
        val_real = LUT[(exp_g.unsqueeze(1).long() << 3) | man_g.unsqueeze(1)]
        val_c = LUT[(centroids.unsqueeze(0).long() << 3) | man_g.unsqueeze(1)] # 结合原始尾数和质心指数
        dist = ((val_real - val_c)**2).sum(-1)

        labels = dist.argmin(dim=1)         # [N]

        # ===== update =====
        new_centroids = []
        for i in range(k):
            mask = (labels == i) # 找到属于第 i 类的所有样本
            if mask.sum() == 0: # 如果该类没有样本
                new_centroids.append(centroids[i])
            else:
                # ⚠️ 只更新 exponent（关键约束）
                new_centroids.append(exp_g[mask].float().mean(dim=0)) 

        centroids = torch.stack(new_centroids)

        # 离散化回 0~15
        centroids = centroids.round().clamp(0, 15)

    return centroids.long(), labels

def kmeans_plus_plus_init(exp_g, k, LUT, H_weight=None):
    """
    exp_g: [N, d]
    返回: 初始 centroids [k, d]
    """
    device = exp_g.device
    N, d = exp_g.shape

    # 转真实值空间（和你原算法一致）
    val_real = LUT[(exp_g.long() << 3)].float()

    centroids = torch.empty((k, d), device=device)

    # 1️⃣ 随机选第一个
    idx = torch.randint(0, N, (1,), device=device)
    centroids[0] = exp_g[idx]

    # 记录每个点到最近centroid的距离
    closest_dist = None

    for i in range(1, k):

        val_c = LUT[(centroids[:i].long().unsqueeze(0) << 3)]

        diff = val_real.unsqueeze(1) - val_c
        if H_weight is not None:
            dist = (diff ** 2 * H_weight.unsqueeze(1)).sum(-1)
        else:
            dist = (diff ** 2).sum(-1)

        min_dist, _ = dist.min(dim=1)

        if closest_dist is None:
            closest_dist = min_dist
        else:
            closest_dist = torch.minimum(closest_dist, min_dist)

        # 2️⃣ 按距离^2采样
        prob = closest_dist + 1e-8
        prob = prob / prob.sum()

        idx = torch.multinomial(prob, 1)
        centroids[i] = exp_g[idx]

    return centroids


def weighted_kmeans_exp_v2(exp_g, k=16, H_weight=None, scale=None, iters=5):
    """
    exp_g: [N, d]
    H_weight: [N, d]  (element-wise Hessian weight)
    """
    device = exp_g.device
    N, d = exp_g.shape

    LUT = FP8_E4M3_LUT.to(exp_g.device)

    # ===== [NEW] 融合 scale 到 H =====
    if scale is not None:
        # scale: [N, 1] 或 [N, d]
        if scale.dim() == 2 and scale.shape[1] == 1:
            scale = scale.expand_as(exp_g)   # broadcast到每个元素

        if H_weight is not None:
            H_eff = H_weight * (scale ** 2)
        else:
            H_eff = scale ** 2
    else:
        H_eff = H_weight
    
    # ===== KMeans++ 初始化（替换原 random init）=====
    centroids = kmeans_plus_plus_init(exp_g, k, LUT, H_eff)

    for _ in range(iters):
        val_real = LUT[(exp_g.long() << 3)]
        val_c = LUT[(centroids.long().unsqueeze(0) << 3)]

        diff = val_real.unsqueeze(1) - val_c

        if H_eff is not None:
            dist = (diff ** 2 * H_eff.unsqueeze(1)).sum(-1)
        else:
            dist = (diff ** 2).sum(-1)

        labels = dist.argmin(dim=1)

        # ===== vectorized update（关键优化版）=====
        centroids_new = torch.zeros_like(centroids)
        weight_sum = torch.zeros_like(centroids)

        for j in range(d):
            wj = H_eff[:, j] if H_eff is not None else torch.ones(N, device=device)

            centroids_new[:, j].index_add_(0, labels, exp_g[:, j].float() * wj)
            weight_sum[:, j].index_add_(0, labels, wj)

        mask = weight_sum > 0
        centroids[mask] = centroids_new[mask] / weight_sum[mask]

        # 离散化
        centroids = centroids.round().clamp(0, 15)

    return centroids.long(), labels

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
    
    def vq(self, Q, all_scales, group_size=4, k=16, row_group_size=1):
        """
        Q: dense quantized tensor with shape [out_features, G, group_size]
        all_scales: per-group scales with shape [out_features, G]
        group_size: VQ的分组大小
        k: VQ的码本大小
        row_group_size: 相邻多少行共用一个VQ码本
        """
        assert row_group_size > 0
        out_features, G, group_size = Q.shape
        in_features = G * group_size
        device = Q.device

        if all_scales.dim() == 2:
            scale_dense = all_scales.unsqueeze(-1).expand(-1, -1, group_size)
        else:
            scale_dense = all_scales
        assert scale_dense.shape == Q.shape

        # 还原到 FP8 空间
        fp8_val = Q / scale_dense

        H_diag = self.H_diag
        if H_diag is not None:
            H_dense = H_diag.view(1, G, group_size).expand(out_features, -1, -1)
        else:
            H_dense = None

        sign, exp, man = fp8_e4m3_decompose(fp8_val)

        vq_group_span = 36
        if G % vq_group_span != 0:
            raise ValueError(f"G={G} is not divisible by vq_group_span={vq_group_span}")

        exp_q = torch.empty_like(exp)
        for row_start in range(0, out_features, row_group_size):
            row_end = min(row_start + row_group_size, out_features)
            for g_start in range(0, G, vq_group_span):
                g_end = g_start + vq_group_span
                block_exp = exp[row_start:row_end, g_start:g_end, :].reshape(-1, group_size)
                block_H = H_dense[row_start:row_end, g_start:g_end, :].reshape(-1, group_size)

                # ===== [NEW] block scale =====
                block_scale = scale_dense[row_start:row_end, g_start:g_end, :]
                block_scale = block_scale.reshape(-1, group_size)[:, :1]

                centroids, labels = weighted_kmeans_exp_v2(
                    block_exp,
                    k=k,
                    H_weight=block_H,
                    scale = block_scale,
                    iters=5
                )
                exp_q[row_start:row_end, g_start:g_end, :] = centroids[labels].view(
                    row_end - row_start, vq_group_span, group_size
                )

        idx = (exp_q << 3) | man  # [out, G, d]
        lut = FP8_E4M3_LUT.to(device)  # [128]
        val = lut[idx]  # 正数
        val = torch.where(sign.bool(), -val, val)
        val = val * scale_dense

        return val.reshape(out_features, in_features)

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, prunen=0, prunem=0, plot=False, vq_dim=4, codebook_size=16, row_group_size=1
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
        self.H_diag = H_diag
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # if static_groups:
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
        # if prunen != 0 and prunem != 0:
        #     print(f"Applying post-quantization wanda {prunen}:{prunem} pruning.")
        #     out_features, in_features = Q.shape

        #     act_norm = torch.sqrt(H_diag + 1e-8)  # shape: (in_features,)
        #     if actorder:
        #         act_norm = act_norm[invperm]

        #     W_metric = torch.abs(Q) * act_norm.view(1, -1)
            
        #     # 针对 N:M，通常在输入特征维度（in_features）进行分组
        #     if in_features % prunem == 0:
        #         # 1. 重塑形状为 (out_features, 组数, M)
        #         W_temp = Q.view(out_features, -1, prunem)
        #         M_temp = W_metric.view(out_features, -1, prunem)

        #         # 2. 找到每组中绝对值最大的前 N 个元素的索引
        #         # topk 会返回前 prunen 个最大值的索引
        #         _, topk_indices = torch.topk(M_temp, prunen, dim=2, largest=True)
                
        #         # 3. 创建掩码并应用
        #         mask = torch.zeros_like(W_temp, dtype=torch.bool)
        #         mask.scatter_(2, topk_indices, True)

        #         pruned_elements = torch.where(~mask, W_temp, torch.zeros_like(W_temp))

        #         """
        #         # ============ MEAN ============
        #         # # 4. 计算被剪掉部分的平均值
        #         # pruned_sum = torch.sum(pruned_elements, dim=2, keepdim=True)

        #         # # 每组被剪掉元素的个数
        #         # num_pruned = prunem - prunen

        #         # # 计算均值
        #         # pruned_mean = pruned_sum / num_pruned
                
        #         # # 5. 均值填充：Top-N 位置保留原值，非 Top-N 位置替换为均值
        #         # W_final = torch.where(mask, W_temp, pruned_mean)

        #         # ============ ZERO ============
        #         # # 5. 0填充
        #         # W_final = torch.where(mask, W_temp, torch.zeros_like(W_temp))

        #         # ============ ACTIVATION-AWARE REPLACEMENT ============
        #         # # 4 activation-aware replacement
        #         # act2 = act_norm.view(1, -1, prunem)

        #         # pruned = ~mask

        #         # num = torch.sum(W_temp * act2 * pruned, dim=2, keepdim=True)
        #         # den = torch.sum(act2 * pruned, dim=2, keepdim=True) + 1e-8 

        #         # replacement = num / den

        #         # # 5 fill
        #         # W_final = torch.where(mask, W_temp, replacement)

        #         # ============ SIGN-AWARE MEAN REPLACEMENT ============
        #         # pos_mask = (pruned_elements > 0)
        #         # neg_mask = (pruned_elements < 0)
        #         # zero_mask = (pruned_elements == 0)
        #         # pos_count = torch.sum(pos_mask, dim=2, keepdim=True).clamp(min=1)
        #         # neg_count = torch.sum(neg_mask, dim=2, keepdim=True).clamp(min=1)
        #         # pos_mean = torch.sum(W_temp * pos_mask, dim=2, keepdim=True) / pos_count
        #         # neg_mean = torch.sum(W_temp * neg_mask, dim=2, keepdim=True) / neg_count

        #         # replacement = torch.zeros_like(W_temp)
        #         # replacement = torch.where(pos_mask, pos_mean, replacement)
        #         # replacement = torch.where(neg_mask, neg_mean, replacement)

        #         # W_final = torch.where(mask, W_temp, replacement)
        #         """

        #         # ============ SIGN-AWARE MEAN REPLACEMENT ============
        #         replacement = torch.zeros_like(W_temp)

        #         pos_mask = ~mask & (pruned_elements > 0)
        #         neg_mask = ~mask & (pruned_elements < 0)

        #         all_scales = torch.cat([group.scale for group in groups], dim=1) # (in_features, out_features / groupsize)
        #         assert(all_scales.shape[0] == out_features)
        #         assert(all_scales.shape[1] == (in_features // prunem))
        #         all_scales = all_scales.unsqueeze(2).expand(-1, -1, prunem)
        #         # FIXME: 
        #         assert(prunem == groupsize)
        #         epsilon = 0.5 * all_scales

        #         replacement = torch.where(pos_mask, epsilon, replacement)
        #         replacement = torch.where(neg_mask, -epsilon, replacement)

        #         # ==================== EXPONENT VQ ====================
        #         if groupsize != -1:
        #             print("Applying exponent VQ...")
        #             W_vq = self.vq(W_temp, mask, prunem, prunen, all_scales, vq_dim, codebook_size, row_group_size)
        #         # ====================================================

        #         W_final = torch.where(mask, W_vq.view(out_features, -1, prunem), replacement)

        #         # --- 新增：FP4 比特打印逻辑 (调试用) ---
        #         if plot:
        #         # 提取被剪枝位置（即 mask 为 False 的位置）的值
        #         # 为了观察 FP4 比特，我们需要除以 scale 还原到量化空间
        #             current_scale = groups[0].scale

        #             if current_scale.dim() == 2:
        #                 # 扩展 scale 维度到 (out_features, 1, 1) 以匹配 (out_features, groups, prunem)
        #                 scale_reshaped = current_scale.unsqueeze(2)
        #             else:
        #                 scale_reshaped = current_scale

        #             # TODO: to plot original(not compensated) pruned weights, here should be W_temp
        #             fp4_query_vals = (W_temp / scale_reshaped)
                    
        #             # 获取比特分解
        #             s, e, m = fp4_e2m1_decompose(fp4_query_vals)
                    
        #             print(f"\n[FP4 Bits for Pruned Elements (Replaced by Mean) | {prunen}:{prunem}]")
        #             # 打印前 32xM 范围内的结构
        #             rows_to_print = min(32, out_features)
        #             groups_to_print = W_final.shape[1]

        #             for r in range(rows_to_print):
        #                 group_bits = []
        #                 for i in range(prunem):
        #                     is_topn = mask[r, 0, i]
        #                     if not is_topn:
        #                         # 被剪枝的位置，现在显示的是均值的 FP4 比特
        #                         bitstr = f"{s[r, 0, i].item()}-{e[r, 0, i].item():02b}-{m[r, 0, i].item()}"
        #                         group_bits.append(f"{bitstr}")
        #                     else:
        #                         # 保留的 Top-N 位置
        #                         group_bits.append("   .  ")
                        
        #                 # 每个 group 打印完后直接输出并换行
        #                 print(" ".join(group_bits))

        #         # 6. 还原回原始二维形状
        #         Q = W_final.view(out_features, in_features)
        #     else:
        #         print(f"Warning: in_features({in_features}) is not divisible by {prunem}. Skipping N:M.")
        # ================================================================

        print("Applying exponent VQ...")
        print("Groupsize:", groupsize)
        out_features, in_features = Q.shape
        W_temp = Q.view(out_features, -1, groupsize)
        all_scales = torch.cat([group.scale for group in groups], dim=1) # (in_features, out_features / groupsize)
        if groupsize != -1:
            W_vq = self.vq(W_temp, all_scales, vq_dim, codebook_size, row_group_size)
        Q = W_vq.view(out_features, in_features)

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
