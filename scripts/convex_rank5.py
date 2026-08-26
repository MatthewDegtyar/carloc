"""Rank-<=5 equivalent of RAFT convex upsampling.

The original builds a rank-7 tensor, `mask.view(N, 1, 9, f, f, H, W)`, which Core
ML rejects outright -- it caps tensors at rank 5. Nothing about the operation
needs rank 7 though: the two `f` axes are only ever recombined into the output
grid at the end, which is precisely what `pixel_shuffle` does.

Keeping the 3x3 neighbourhood (9) and the upsample block (f*f) as separate rank-5
axes, then folding the block into space with pixel_shuffle, computes the same
values with a maximum rank of 5.
"""
import torch
import torch.nn.functional as F


def original(flow, mask, factor):
    N, D, H, W = flow.shape
    m = mask.view(N, 1, 9, factor, factor, H, W)
    m = torch.softmax(m, dim=2)
    up = F.unfold(flow, [3, 3], padding=1).view(N, D, 9, 1, 1, H, W)
    up = torch.sum(m * up, dim=2)
    up = up.permute(0, 1, 4, 2, 5, 3)
    return up.reshape(N, D, factor * H, factor * W)


def rank5(flow, mask, factor):
    N, D, H, W = flow.shape
    ff = factor * factor
    m = torch.softmax(mask.view(N, 9, ff, H, W), dim=1)          # rank 5
    up = F.unfold(flow, [3, 3], padding=1).view(N, D, 9, H, W)   # rank 5
    outs = []
    for d in range(D):
        # (N,9,ff,H,W) * (N,9,1,H,W) -> sum over the 9 neighbours -> (N,ff,H,W)
        outs.append((m * up[:, d].unsqueeze(2)).sum(dim=1))
    out = torch.stack(outs, dim=1)                               # (N,D,ff,H,W)
    # pixel_shuffle maps channel c*r*r + i*r + j -> pixel (h*r+i, w*r+j), which is
    # exactly the permute-and-reshape the original ends with.
    return F.pixel_shuffle(out.reshape(N, D * ff, H, W), factor)


if __name__ == "__main__":
    torch.manual_seed(0)
    for N, D, H, W, factor in ((1, 1, 7, 11, 8), (2, 3, 5, 5, 4), (1, 1, 4, 6, 2)):
        flow = torch.randn(N, D, H, W)
        mask = torch.randn(N, 9 * factor * factor, H, W)
        a, b = original(flow, mask, factor), rank5(flow, mask, factor)
        same = torch.allclose(a, b, atol=1e-6)
        print(f"N={N} D={D} {H}x{W} factor={factor}: shapes {tuple(a.shape)}=={tuple(b.shape)} "
              f"max|diff|={ (a-b).abs().max().item():.2e}  identical={same}")
