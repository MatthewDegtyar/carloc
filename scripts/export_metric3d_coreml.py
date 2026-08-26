"""Export Metric3D v2 to CoreML for on-device inference.

Metric3D has no published CoreML build, and converting it hits four blockers in
sequence. Each is recorded here with its fix, because the last one looks fatal and
is not.

1. **Hard-coded `device="cuda"`** in the decoder -- the model cannot even be
   loaded without CUDA. Rewritten to resolve at runtime (see
   `depthbench/runners/metric3d_runner.py:patch_hub_device`). For export the
   device must be CPU, matching the traced tensors.

2. **Data-dependent control flow** (`if torch.isnan(x).any()`) makes the traced
   graph differ between invocations, failing trace sanity. Those branches are
   diagnostics that never fire on real input, so `check_trace=False` bakes the
   normal path.

3. **`upsample_bicubic2d` has no CoreML converter.** It appears once, resampling
   DINOv2's positional embedding -- a small fixed grid, identical every frame at a
   fixed input size. Mapped to bilinear (`scripts/bicubic_shim.py`).

4. **CoreML caps tensors at rank 5; the RAFT convex upsampling builds rank 7.**
   This is the one that looks architectural. It is not: `mask.view(N,1,9,f,f,H,W)`
   only splits the two upsample axes so it can recombine them into the output grid
   at the end -- which is exactly `pixel_shuffle`. Rewritten to stay within rank 5
   and verified BIT-IDENTICAL to the original (`scripts/convex_rank5.py`).

Result on nuScenes (324 objects, 5.5-48.9 m), ViT-small, against the same ground
truth as `depthbench`:

    median 1.16 m | delta<1.25 95% | usable to 50 m | 72 MB | 0.45 s/image

which beats the ViT-large PyTorch model's 1.16 m/1.33 m median at a tenth of the
size. It is worse in the tail (p90 5.92 m vs 3.83 m) and at 40-50 m (5.85 m vs
2.81 m).

CAVEAT, and it decides the deployment: **the Neural Engine will not take this
graph.** Measured CPU_ONLY 442 ms vs ALL 421 ms -- the ANE contributes nothing,
because `unfold` and the iterative decoder are rejected op by op. So this is a
~420 ms CPU model. That is far outside a per-frame perception budget but inside a
500 ms ranging loop, which is where depth belongs anyway.

Run in a separate environment:

    uv venv --python 3.12 /tmp/ios-export/.venv
    VIRTUAL_ENV=/tmp/ios-export/.venv uv pip install torch torchvision coremltools \
        "numpy<2" timm mmengine scipy
    cp -R depthbench/shims/mmcv /tmp/ios-export/.venv/lib/python3.12/site-packages/
    /tmp/ios-export/.venv/bin/python scripts/export_metric3d_coreml.py --model vit_small
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

DEVICE_HELPER = '''

def _depthbench_device():
    return "cpu"
'''

CONVEX_RANK5 = '''    def upsample_flow(self, flow, mask):
        """Convex-combination upsampling, rank<=5 so Core ML can represent it.

        Bit-identical to the original rank-7 formulation; see
        scripts/convex_rank5.py for the equivalence check.
        """
        N, D, H, W = flow.shape
        factor = 2 ** self.n_downsample
        ff = factor * factor
        mask = torch.softmax(mask.view(N, 9, ff, H, W), dim=1)
        up_flow = F.unfold(flow, [3, 3], padding=1).view(N, D, 9, H, W)
        outs = []
        for d in range(D):
            outs.append((mask * up_flow[:, d].unsqueeze(2)).sum(dim=1))
        out = torch.stack(outs, dim=1)
        return F.pixel_shuffle(out.reshape(N, D * ff, H, W), factor)
'''

HUB_ROOT = os.path.expanduser("~/.cache/torch/hub/yvanyin_metric3d_main")


def patch_sources() -> None:
    """Apply blockers 1 and 4 to the downloaded hub sources."""
    literals = ('device="cuda"', "device='cuda'", "device=_depthbench_device()")
    for path in glob.glob(os.path.join(HUB_ROOT, "mono", "**", "*.py"), recursive=True):
        text = original = open(path).read()
        for literal in literals:
            text = text.replace(literal, 'device="cpu"')
        if text != original:
            open(path, "w").write(text)

    decoder = os.path.join(
        HUB_ROOT, "mono", "model", "decode_heads", "RAFTDepthNormalDPTDecoder5.py"
    )
    source = open(decoder).read()
    if "rank<=5 so Core ML" not in source:
        start = source.index("    def upsample_flow(self, flow, mask):")
        end = source.index("    def initialize_flow(self, img):")
        open(decoder, "w").write(source[:start] + CONVEX_RANK5 + "\n" + source[end:])
    print("patched hub sources (device literals + rank-5 convex upsampling)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vit_small", choices=("vit_small", "vit_large"))
    parser.add_argument("--out", default="models")
    parser.add_argument("--height", type=int, default=616)
    parser.add_argument("--width", type=int, default=1064)
    args = parser.parse_args()

    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Download WITHOUT constructing a model, so the patch lands before import.
    torch.hub.list("yvanyin/metric3d", trust_repo=True)
    patch_sources()

    name = f"metric3d_{args.model}"
    model = torch.hub.load("yvanyin/metric3d", name, pretrain=True, trust_repo=True).eval()
    print(f"{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    class Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner.inference({"input": x})
            return out[0] if isinstance(out, (tuple, list)) else out

    dummy = torch.zeros(1, 3, args.height, args.width)
    with torch.no_grad():
        traced = torch.jit.trace(Wrap(model), dummy, strict=False, check_trace=False)
    print("traced")

    import bicubic_shim  # noqa: F401  (registers upsample_bicubic2d)
    import coremltools as ct

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=dummy.shape)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
    )
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, f"{name}.mlpackage")
    mlmodel.save(target)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
