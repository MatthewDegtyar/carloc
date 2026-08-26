"""Export Metric3D for iOS: image input, aspect-matched to the camera.

Two fixes over the plain export, both about what the phone actually hands you.

**1. Image input, not a float array.** ARKit gives a `CVPixelBuffer`. A model with
an `MLMultiArray` input forces a conversion of several MB per frame, on the CPU,
before inference can start -- pure overhead that scales with resolution. An
`ImageType` input takes the buffer directly. Apple's DAv2 build does this, which is
part of why it benchmarks at 25 ms.

Metric3D expects ImageNet-normalised input, and Core ML's `ImageType` can only
apply a single scalar scale with a per-channel bias -- it cannot express division
by three different per-channel standard deviations. So the normalisation is baked
into the model instead, which is exact rather than a 2% approximation.

**2. Aspect-matched input size.** ARKit's frame is 1920x1440, aspect 0.75. The
benchmark config 448x784 has aspect 0.571, so letterboxing a phone frame into it
leaves **24% of the tokens on grey bars**. 504x672 is 4:3 exactly, costs 1728
tokens against 1792, and spends every one of them on image.

Cost is quadratic in tokens, so this is close to free accuracy.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# 4:3 sizes, both dimensions divisible by the ViT patch size of 14.
IOS_SIZES = {
    "336x448": (336, 448),    # 768 tokens
    "420x560": (420, 560),    # 1200 tokens
    "504x672": (504, 672),    # 1728 tokens -- matches 448x784's cost, no padding
    "546x728": (546, 728),    # 2028 tokens
}

# ImageNet statistics Metric3D was trained with.
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]


class NormalisedMetric3D(torch.nn.Module):
    """Wraps the model so it accepts raw 0-255 RGB and normalises internally.

    Core ML's ImageType exposes one scalar `scale` and a per-channel `bias`, which
    cannot represent per-channel division. Doing it inside the graph keeps the
    arithmetic exact and lets the Swift side pass a pixel buffer with no
    preprocessing at all.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = (image - self.mean) / self.std
        out = self.model.inference({"input": x})
        return out[0] if isinstance(out, (tuple, list)) else out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="504x672", choices=sorted(IOS_SIZES))
    parser.add_argument("--model", default="vit_small", choices=("vit_small", "vit_large"))
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from export_metric3d_coreml import patch_sources

    torch.hub.list("yvanyin/metric3d", trust_repo=True)
    patch_sources()
    try:  # chunked attention: exact, and cheaper once the attention matrix is large
        import chunk_attn  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"(chunked attention not applied: {exc})")

    inner = torch.hub.load("yvanyin/metric3d", f"metric3d_{args.model}", pretrain=True,
                           trust_repo=True).eval()
    model = NormalisedMetric3D(inner).eval()

    h, w = IOS_SIZES[args.size]
    print(f"{args.model} @ {h}x{w} = {(h // 14) * (w // 14)} tokens, aspect {h / w:.3f}")

    dummy = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=False, check_trace=False)

    import bicubic_shim  # noqa: F401
    import coremltools as ct

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, h, w),
                             color_layout=ct.colorlayout.RGB, scale=1.0, bias=[0.0, 0.0, 0.0])],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
    )
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, f"metric3d_{args.model}_ios_{args.size}.mlpackage")
    mlmodel.save(target)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
