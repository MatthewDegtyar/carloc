"""Metric3D v2 (ViT-large) via torch.hub.

Run twice, as two entries: once with the manifest's true intrinsics and once with
the model's defaults. Metric3D canonicalises the image using the focal length, so
the difference between those runs is the value of knowing your camera -- which is
the interesting question, and it is invisible unless both are measured.
"""

import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402

INPUT_SIZE = (616, 1064)  # the ViT canonical input Metric3D expects
CANONICAL_FOCAL = 1000.0
DEFAULT_FOCAL = 1000.0
"""Stand-in focal used by the default-intrinsics run.

Metric3D's own examples use ~1000 px for an unknown camera. The nuScenes CAM_FRONT
focal is ~1253 px, so this run is wrong by about 25% -- which is roughly how wrong
you are when you point a metric model at a camera you have not calibrated."""


def prepare(image: np.ndarray, focal: float):
    """Metric3D's canonical resize-and-pad, plus the scale needed to undo it."""
    import torch

    h, w = image.shape[:2]
    scale = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))

    pad_h, pad_w = INPUT_SIZE[0] - new_h, INPUT_SIZE[1] - new_w
    top, left = pad_h // 2, pad_w // 2
    padded = np.pad(
        resized, ((top, pad_h - top), (left, pad_w - left), (0, 0)),
        mode="constant", constant_values=123,
    )
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    tensor = torch.from_numpy(((padded - mean) / std).transpose(2, 0, 1)).float()[None]
    # Metric3D predicts depth in a canonical space where the focal length is
    # CANONICAL_FOCAL. A longer real focal makes an object of given apparent size
    # further away, so real depth = canonical depth * (f_effective / f_canonical),
    # with f_effective being the focal after the resize.
    canonical_to_real = (focal * scale) / CANONICAL_FOCAL
    return tensor, (top, left, new_h, new_w), canonical_to_real


DEVICE_HELPER = """

def _depthbench_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
"""

CUDA_LITERALS = (
    'device="cuda"',
    "device='cuda'",
)


def patch_hub_device() -> int:
    """Make Metric3D's decoder device-agnostic.

    `RAFTDepthNormalDPTDecoder5` builds its depth bins and mesh grid with a literal
    `device="cuda"`, so the model cannot run anywhere without CUDA no matter where
    its weights were placed.

    This is a device-placement fix, not a model change: the tensors created are
    identical, only their location differs. The resolver must agree with the one
    the runner uses -- resolving to "cpu" while the weights sit on "mps" fails on
    the first op that touches both.

    Idempotent by design. It runs before every load and must therefore cope with
    a cache it has already rewritten, otherwise a later correction silently does
    nothing because the original literal is gone.
    """
    import glob

    patched = 0
    root = os.path.expanduser("~/.cache/torch/hub/yvanyin_metric3d_main")
    for path in glob.glob(os.path.join(root, "mono", "**", "*.py"), recursive=True):
        with open(path) as handle:
            text = handle.read()
        original = text
        for literal in CUDA_LITERALS:
            if literal in text:
                patched += text.count(literal)
                text = text.replace(literal, "device=_depthbench_device()")
        # Repair any earlier substitution that resolved differently.
        stale = 'device=("cuda" if __import__("torch").cuda.is_available() else "cpu")'
        if stale in text:
            patched += text.count(stale)
            text = text.replace(stale, "device=_depthbench_device()")
        if text == original:
            continue
        if "def _depthbench_device" not in text:
            # PREPENDED, not appended. One of the patched sites is a default
            # argument (`def create_mesh_grid(..., device=...)`), which Python
            # evaluates at class-definition time -- so a helper defined at the
            # bottom of the file does not exist yet when the class is built.
            lines = text.split("\n")
            insert_at = 0
            for index, line in enumerate(lines[:40]):
                if line.startswith("from __future__"):
                    insert_at = index + 1
            text = "\n".join(lines[:insert_at]) + DEVICE_HELPER + "\n".join(lines[insert_at:])
        with open(path, "w") as handle:
            handle.write(text)
    return patched


def main():
    args = _common.parse_args()
    use_real_K = os.environ.get("DEPTHBENCH_REAL_K", "1") == "1"
    manifest = _common.load_manifest(args.manifest)
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    device = args.device or _common.pick_device()

    import torch

    # Fetch the repo WITHOUT building a model, so the device patch lands on the
    # sources before they are imported. `hub.load` downloads and constructs in one
    # call, so patching before it is a no-op on a cold cache and patching after is
    # too late -- the module is already imported.
    #
    # trust_repo: torch.hub otherwise prompts on stdin, which in a captured
    # subprocess is an EOFError rather than a prompt.
    torch.hub.list("yvanyin/metric3d", trust_repo=True)
    patched = patch_hub_device()
    model = torch.hub.load(
        "yvanyin/metric3d", "metric3d_vit_large", pretrain=True, trust_repo=True
    )
    model.to(device).eval()

    focal_note = "per-image from manifest" if use_real_K else f"{DEFAULT_FOCAL:.0f} px fixed"
    predictions, started = [], time.time()
    for sample in samples:
        image = np.asarray(Image.open(sample["image"]).convert("RGB"))
        focal = float(sample["K"][1][1]) if use_real_K else DEFAULT_FOCAL
        tensor, (top, left, new_h, new_w), canonical_to_real = prepare(image, focal)
        with torch.no_grad():
            depth, _, _ = model.inference({"input": tensor.to(device)})
        depth = depth.squeeze().cpu().numpy()
        # Undo pad, then undo the canonical-focal transform to get metres.
        depth = depth[top : top + new_h, left : left + new_w]
        # MULTIPLY. Dividing here inflates every prediction by 1/scale^2 -- about
        # 1.44x on this camera -- while leaving rank correlation near 1.0, so the
        # model looks badly calibrated rather than mis-scaled by the harness.
        depth = np.asarray(depth, dtype=np.float64) * canonical_to_real
        depth = np.clip(depth, 0, 300)
        for obj in sample["objects"]:
            predictions.append({
                "obj_id": obj["obj_id"], "image": sample["image"],
                "pred_depth_m": _common.sample_depth(
                    depth, obj["bbox"], (sample["width"], sample["height"])
                ),
            })
    _common.write_result(
        args.out, "Metric3D v2", "real intrinsics" if use_real_K else "default intrinsics",
        predictions, (time.time() - started) / max(len(samples), 1), device,
        notes=(
            f"focal used: {focal_note}; mmcv shim in use (logging only); "
            f"{patched} hard-coded cuda device literals patched for CPU/MPS"
        ),
    )


if __name__ == "__main__":
    main()
