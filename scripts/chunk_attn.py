"""Chunk the ViT attention so its working set stays under the ANE cliff.

Measured: the ANE speedup falls from 2.76x to 1.15x as the attention matrix grows
from 3.7 MB to 134 MB. Nothing is rejected -- the working set simply outgrows
on-chip memory and spills.

Chunking over the query axis computes the same values (each query row's softmax is
independent of every other row) while capping the live attention block at
chunk x N instead of N x N. At chunk=512 and N=3344 that is 20 MB rather than 134.

Mathematically identical, not an approximation: softmax is applied along the key
axis, which every chunk keeps whole.
"""
import os

PATH = os.path.expanduser(
    "~/.cache/torch/hub/yvanyin_metric3d_main/mono/model/backbones/ViT_DINO_reg.py")

CHUNKED = '''    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        chunk = 512
        kt = k.transpose(-2, -1)
        outs = []
        for start in range(0, N, chunk):
            qi = q[:, :, start:start + chunk, :]
            ai = qi @ kt
            if attn_bias is not None:
                ai = ai + attn_bias[:, :, start:start + chunk, :N]
            ai = ai.softmax(dim=-1)
            outs.append(ai @ v)
        x = torch.cat(outs, dim=2).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
'''

source = open(PATH).read()
if "chunk = 512" in source:
    print("already chunked")
else:
    anchor = (
        "    def forward(self, x: Tensor, attn_bias=None) -> Tensor:\n"
        "        B, N, C = x.shape"
    )
    start = source.index(anchor)
    # keep everything after the Attention.forward body up to MemEffAttention
    tail_start = source.index("\n\nclass MemEffAttention(Attention):")
    open(PATH, "w").write(source[:start] + CHUNKED + source[tail_start:])
    print("patched Attention.forward with chunked attention")
