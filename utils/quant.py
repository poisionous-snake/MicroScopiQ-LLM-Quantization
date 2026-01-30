import torch
import torch.nn as nn

class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('scale', torch.zeros(shape))
        self.max_representable = 6.0 # E2M1 format

    def configure(self, bits=4, groupsize=32):
        self.bits = bits
        self.groupsize = groupsize

    def find_params(self, x, weight=False):
        max_vals, _ = torch.max(torch.abs(x), dim=1, keepdim=True)
        exponents = torch.ceil(torch.log2(max_vals / self.max_representable + 1e-12))
        exponents = torch.clamp(exponents, min=-(2**7), max=2**7-1)
        self.scale = torch.pow(2.0, exponents)

    def apply_mxfp4_mapping(self, x_norm):
        values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x_norm.device)
        sign = torch.sign(x_norm)
        abs_x = torch.abs(x_norm)

        # round to nearest
        midpoints = (values[:-1] + values[1:]) / 2
        indices = torch.bucketize(abs_x, midpoints)
        assert indices.max() < len(values)

        return values[indices] * sign

    def quantize(self, x):
        if self.ready():
            x_norm = x / self.scale
            x_q = self.apply_mxfp4_mapping(x_norm)
            return x_q * self.scale
        return x

    def ready(self):
        return torch.all(self.scale != 0)