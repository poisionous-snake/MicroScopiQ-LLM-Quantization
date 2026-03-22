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

def kmeans_exp_vq(exp_g, man_g, k=16, iters=10):
    """
    exp_g: [N, d]
    man_g: [N, d]
    """
    N = exp_g.shape[0]

    # === 初始化 centroid（从数据中采样）===
    unique_exp = torch.unique(exp_g, dim=0) # [U, d]
    print(f"Unique exponent patterns: {unique_exp.shape[0]}")
    idx = torch.randperm(unique_exp.shape[0])[:k]
    centroids = unique_exp[idx].float() # [k, d]

    for _ in range(iters):
        # # === SIMPLEST DIST ===
        # dist = ((exp_g.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(-1)  # [N, k]
        
        # # === 2^exp DIST ===
        # dist = ((2 ** (exp_g.unsqueeze(1) - 1)  - 2 ** (centroids.unsqueeze(0) - 1)) ** 2).sum(-1)  # [N, k]
        
        # === +man LUT DIST ===
        LUT = FP4_E2M1_LUT.to(exp_g.device) # [16]
        val_real = LUT[(exp_g.unsqueeze(1).long() << 1) | man_g.unsqueeze(1)]
        val_c = LUT[(centroids.unsqueeze(0).long() << 1) | man_g.unsqueeze(1)] # 结合原始尾数和质心指数
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

        # 离散化回 0~3
        centroids = centroids.round().clamp(0, 3)

    return centroids.long(), labels

def topk_exp_vq(exp_g, man_g, k=16):
    """
    exp_g: [N, d] 原始指数张量
    k: 码本大小 (Top-K 模式数量)
    """
    device = exp_g.device
    N, d = exp_g.shape

    # 1. 将 d 维指数映射为唯一的 Key (假设指数范围 0-3, 即 2 bits)
    # 这样可以将 [N, d] 的向量转换为 [N] 的一维整数，方便统计频率
    keys = torch.zeros(N, device=device, dtype=torch.long)
    for i in range(d):
        keys += exp_g[:, i].long() * (4 ** (d - 1 - i))

    # 2. 统计所有模式出现的频率
    unique_keys, counts = torch.unique(keys, return_counts=True)
    
    # 3. 选取出现次数最多的前 K 个模式作为码本
    actual_k = min(k, unique_keys.size(0))
    topk_counts, topk_indices = torch.topk(counts, actual_k)
    codebook_keys = unique_keys[topk_indices]

    # 4. 将选中的 Key 还原为 d 维指数向量 (Centroids)
    centroids = []
    for key in codebook_keys:
        pattern = []
        tmp = key.item()
        for _ in range(d):
            pattern.append(tmp % 4)
            tmp //= 4
        centroids.append(pattern[::-1])
    
    centroids = torch.tensor(centroids, device=device, dtype=torch.long) # [actual_k, d]

    # 5. 分配 (Assignment): 将原始数据映射到最近的码本项
    # 由于指数是离散的，我们直接计算每个样本 keys 与 codebook_keys 的距离
    # 这里使用简单的欧式距离或曼哈顿距离即可，或者直接在 Key 空间找最接近的值
    # 为了严谨，我们计算值空间距离 (类似于 K-means 的计算)
    
    # 扩展维度进行广播计算: [N, 1, d] vs [1, actual_k, d]
    dist = ((exp_g.unsqueeze(1).float() - centroids.unsqueeze(0).float()) ** 2).sum(-1)
    # # === +man LUT DIST ===
    # LUT = FP4_E2M1_LUT.to(exp_g.device) # [16]
    # val_real = LUT[(exp_g.unsqueeze(1).long() << 1) | man_g.unsqueeze(1)]
    # val_c = LUT[(centroids.unsqueeze(0).long() << 1) | man_g.unsqueeze(1)] # 结合原始尾数和质心指数
    # dist = ((val_real - val_c)**2).sum(-1)
    labels = dist.argmin(dim=1)

    return centroids, labels

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
    
    def vq(self, Q, mask, m, n, all_scales, group_size=4, k=16):
        """
        Q: 量化剪枝后的向量
        groups: scale
        group_size: VQ的分组大小
        k: VQ的码本大小
        """
        out_features, in_features = Q.shape[0], Q.shape[1] * Q.shape[2]
        device = Q.device

        # 还原到FP4空间
        fp4_val = Q / all_scales

        dense_out_feature = (in_features * n) // m
        G = dense_out_feature // group_size
        x = fp4_val.masked_select(mask).view(
            out_features, G, group_size
        )
        print(x[:4][:24])

        # 分解到FP4比特
        sign, exp, man = fp4_e2m1_decompose(x)

        print("G:", G)
        for rows in range(min(4, out_features)):
            x = exp[rows]
            cnt = torch.zeros(G, device=exp.device, dtype=torch.int32)
            cnt_delta = torch.zeros(G, device=exp.device, dtype=torch.int32)
            for i in range(G):
                cnt[i] = x[i, 0] + x[i, 1] * 4 + x[i, 2] * 16 + x[i, 3] * 64
                cnt_delta[i] = x[i, 0] + (x[i, 1] - x[i, 0]) * 4 + (x[i, 2] - x[i, 1]) * 16 + (x[i, 3] - x[i, 2]) * 64
            # 统计cnt中的unique格式
            unique_cnt = torch.unique(cnt)
            unique_cnt_delta = torch.unique(cnt_delta)
            print(f"Unique exponent patterns in groups: {len(unique_cnt)}")
            print(f"Unique exponent delta patterns in groups: {len(unique_cnt_delta)}")

        # === reshape ===
        X_exp = exp.reshape(-1, group_size)
        X_man = man.reshape(-1, group_size)

        # === K-means ===
        centroids, labels = topk_exp_vq(X_exp, X_man, k)
        print("Number of unique centroids:", len(centroids))
        # === 重建 ===
        X_q = centroids[labels]
        exp_q = X_q.view(out_features, G, group_size)

        idx = (exp_q << 1) | man  # [out, G, d]
        lut = FP4_E2M1_LUT.to(device)  # [16]
        val = lut[idx]  # 正数
        val = torch.where(sign.bool(), -val, val)

        # 重构为sparse格式
        sparse_val = torch.zeros((out_features, in_features), device=device)
        sparse_val.masked_scatter_(mask.view(out_features, in_features), val.reshape(out_features, -1))
        val = sparse_val * all_scales.reshape(out_features, in_features)

        print(sparse_val[:4][:32])

        return val

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, prunen=0, prunem=0, plot=False, vq_dim=4, codebook_size=16
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

                pruned_elements = torch.where(~mask, W_temp, torch.zeros_like(W_temp))

                """
                # ============ MEAN ============
                # # 4. 计算被剪掉部分的平均值
                # pruned_sum = torch.sum(pruned_elements, dim=2, keepdim=True)

                # # 每组被剪掉元素的个数
                # num_pruned = prunem - prunen

                # # 计算均值
                # pruned_mean = pruned_sum / num_pruned
                
                # # 5. 均值填充：Top-N 位置保留原值，非 Top-N 位置替换为均值
                # W_final = torch.where(mask, W_temp, pruned_mean)

                # ============ ZERO ============
                # # 5. 0填充
                # W_final = torch.where(mask, W_temp, torch.zeros_like(W_temp))

                # ============ ACTIVATION-AWARE REPLACEMENT ============
                # # 4 activation-aware replacement
                # act2 = act_norm.view(1, -1, prunem)

                # pruned = ~mask

                # num = torch.sum(W_temp * act2 * pruned, dim=2, keepdim=True)
                # den = torch.sum(act2 * pruned, dim=2, keepdim=True) + 1e-8 

                # replacement = num / den

                # # 5 fill
                # W_final = torch.where(mask, W_temp, replacement)

                # ============ SIGN-AWARE MEAN REPLACEMENT ============
                # pos_mask = (pruned_elements > 0)
                # neg_mask = (pruned_elements < 0)
                # zero_mask = (pruned_elements == 0)
                # pos_count = torch.sum(pos_mask, dim=2, keepdim=True).clamp(min=1)
                # neg_count = torch.sum(neg_mask, dim=2, keepdim=True).clamp(min=1)
                # pos_mean = torch.sum(W_temp * pos_mask, dim=2, keepdim=True) / pos_count
                # neg_mean = torch.sum(W_temp * neg_mask, dim=2, keepdim=True) / neg_count

                # replacement = torch.zeros_like(W_temp)
                # replacement = torch.where(pos_mask, pos_mean, replacement)
                # replacement = torch.where(neg_mask, neg_mean, replacement)

                # W_final = torch.where(mask, W_temp, replacement)
                """

                # ============ SIGN-AWARE MEAN REPLACEMENT ============
                replacement = torch.zeros_like(W_temp)

                pos_mask = ~mask & (pruned_elements > 0)
                neg_mask = ~mask & (pruned_elements < 0)

                all_scales = torch.cat([group.scale for group in groups], dim=1) # (in_features, out_features / groupsize)
                assert(all_scales.shape[0] == out_features)
                assert(all_scales.shape[1] == (in_features // prunem))
                all_scales = all_scales.unsqueeze(2).expand(-1, -1, prunem)
                # FIXME: 
                assert(prunem == groupsize)
                epsilon = 0.5 * all_scales

                replacement = torch.where(pos_mask, epsilon, replacement)
                replacement = torch.where(neg_mask, -epsilon, replacement)

                # ==================== EXPONENT VQ ====================
                if groupsize != -1:
                    print("Applying exponent VQ...")
                    W_vq = self.vq(W_temp, mask, prunem, prunen, all_scales, vq_dim, codebook_size)
                # ====================================================

                W_final = torch.where(mask, W_vq.view(out_features, -1, prunem), replacement)

                # --- 新增：FP4 比特打印逻辑 (调试用) ---
                if plot:
                # 提取被剪枝位置（即 mask 为 False 的位置）的值
                # 为了观察 FP4 比特，我们需要除以 scale 还原到量化空间
                    current_scale = groups[0].scale

                    if current_scale.dim() == 2:
                        # 扩展 scale 维度到 (out_features, 1, 1) 以匹配 (out_features, groups, prunem)
                        scale_reshaped = current_scale.unsqueeze(2)
                    else:
                        scale_reshaped = current_scale

                    # TODO: to plot original(not compensated) pruned weights, here should be W_temp
                    fp4_query_vals = (W_temp / scale_reshaped)
                    
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
