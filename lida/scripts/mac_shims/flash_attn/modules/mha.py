"""Cross-attention with flash_attn's parameter names, for machines without CUDA.

UniRig's skin model only uses MHA(embed_dim, num_heads, cross_attn=True). The layers
match flash_attn's Wq / Wkv / out_proj, so the published checkpoint loads unchanged.
"""
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, cross_attn=False, **_):
        super().__init__()
        if not cross_attn:
            raise NotImplementedError("Only the cross-attention form used by UniRig is provided.")
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, x_kv=None, **_):
        B, Nq = x.shape[:2]
        h, d = self.num_heads, self.head_dim
        kv_in = x if x_kv is None else x_kv
        Nk = kv_in.shape[1]
        q = self.Wq(x).reshape(B, Nq, h, d).transpose(1, 2)  # (B, h, Nq, d)
        kv = self.Wkv(kv_in).reshape(B, Nk, 2, h, d)  # flash_attn's packed layout
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(B, Nq, h * d))
