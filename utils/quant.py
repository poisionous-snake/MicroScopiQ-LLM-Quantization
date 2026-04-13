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
    qmax = 448.0
    x_scaled = x_norm.abs().clamp(max=qmax)

    sign = torch.sign(x_norm)
    nonzero_scaled = x_scaled + (x_scaled == 0).to(x_scaled.dtype)
    exp = nonzero_scaled.log2().floor().clamp_(min=-6, max=8)
    man = torch.round(x_scaled / (2 ** exp) * (2 ** 3)) / (2 ** 3)
    x_q = sign * (2 ** exp) * man

    return torch.clamp(x_q, min=-qmax, max=qmax)


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