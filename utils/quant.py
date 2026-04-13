import torch
import torch.nn as nn

def apply_mxfp4_mapping(x_norm):
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=x_norm.device,
        dtype=x_norm.dtype,
    )
    sign = torch.sign(x_norm)
    abs_x = torch.abs(x_norm)

    # round to nearest
    midpoints = (values[:-1] + values[1:]) / 2
    indices = torch.bucketize(abs_x, midpoints)
    assert indices.max() < len(values)

    return values[indices] * sign

def apply_mxfp8_mapping(x_norm):
    # FP8(E4M3) hardcoded LUT (positive values only)
    # index = (exponent << 3) | mantissa
    values = torch.tensor([
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
    ], device=x_norm.device, dtype=x_norm.dtype)
    
    sign = torch.sign(x_norm)
    abs_x = torch.abs(x_norm)

    # round to nearest
    midpoints = (values[:-1] + values[1:]) / 2
    indices = torch.bucketize(abs_x, midpoints)
    assert indices.max() < len(values)

    return values[indices] * sign


def quantize(x, scale, q_bits=4):
    x_norm = x / scale
    if q_bits >= 8:
        x_q = apply_mxfp8_mapping(x_norm)
    else:
        x_q = apply_mxfp4_mapping(x_norm)
    return x_q * scale

class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('scale', torch.zeros(shape))
        self.bits = 4
        self.groupsize = 32
        self.max_representable = 6.0

    def configure(self, bits=4, groupsize=32):
        self.bits = bits
        self.groupsize = groupsize
        self.max_representable = 448.0 if bits >= 8 else 6.0

    def find_params(self, x, weight=False):
        max_vals, _ = torch.max(torch.abs(x), dim=1, keepdim=True)
        # exponents = torch.ceil(torch.log2(max_vals / self.max_representable + 1e-12))
        # exponents = torch.clamp(exponents, min=-(2**7), max=2**7-1)
        # self.scale = torch.pow(2.0, exponents)

        scale = max_vals / self.max_representable
        scale = scale.clamp_(min=1e-6)
        self.scale = scale

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, q_bits=self.bits)
        return x

    def ready(self):
        return torch.all(self.scale != 0)