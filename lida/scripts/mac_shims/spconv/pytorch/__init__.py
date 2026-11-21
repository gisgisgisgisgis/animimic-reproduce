"""The small part of spconv that UniRig's PTv3 encoder uses, in plain torch.

spconv ships no macOS wheels. UniRig only needs SubMConv3d: the output exists at the
input's occupied voxels, and each one sums the kernel over occupied neighbours.

The parameter shapes match the checkpoint: weight is (out, kx, ky, kz, in) and bias is
(out,). The kernel is applied as nn.Conv3d does: out[p] = sum_o w[o] * in[p + o].
This has NOT been compared numerically against the real spconv.
"""
import types

import torch
import torch.nn as nn


class SparseModule(nn.Module):
    pass


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size):
        self.features, self.indices, self.spatial_shape, self.batch_size = features, indices, spatial_shape, batch_size

    def replace_feature(self, features):
        return SparseConvTensor(features, self.indices, self.spatial_shape, self.batch_size)


class SubMConv3d(SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, **_):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, kernel_size, kernel_size, kernel_size, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        idx = x.indices.long()
        feat = x.features
        r = self.kernel_size // 2
        sx, sy, sz = (int(s) + 2 * r + 1 for s in x.spatial_shape)

        def key(i, ox=0, oy=0, oz=0):
            return ((i[:, 0] * sx + i[:, 1] + r + ox) * sy + i[:, 2] + r + oy) * sz + i[:, 3] + r + oz

        base = key(idx)
        order = torch.argsort(base)
        sorted_key = base[order]
        out = torch.zeros(feat.shape[0], self.weight.shape[0], device=feat.device, dtype=feat.dtype)
        for ox in range(-r, r + 1):
            for oy in range(-r, r + 1):
                for oz in range(-r, r + 1):
                    want = key(idx, ox, oy, oz)
                    pos = torch.searchsorted(sorted_key, want).clamp(max=sorted_key.shape[0] - 1)
                    hit = sorted_key[pos] == want
                    w = self.weight[:, ox + r, oy + r, oz + r, :]  # (out, in)
                    out = out + (feat[order[pos]] @ w.T) * hit[:, None].to(feat.dtype)
        if self.bias is not None:
            out = out + self.bias
        return x.replace_feature(out)


def is_spconv_module(module):
    return isinstance(module, SparseModule)


modules = types.SimpleNamespace(is_spconv_module=is_spconv_module)
