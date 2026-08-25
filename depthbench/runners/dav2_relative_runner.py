"""Depth Anything V2 relative, rescaled to metres from one known-size object.

The relative checkpoint emits inverse depth (disparity): larger means nearer. It
is converted with ``metric = scale / value``, the scale coming from the reference
object's pinhole depth. The scorer reports rank correlation against truth, which
is what would expose this convention being wrong -- a negative correlation means
the map was read upside down.
"""

import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402

MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
INVERSE = True


def main():
    args = _common.parse_args()
    manifest = _common.load_manifest(args.manifest)
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]

    from transformers import pipeline

    device = args.device or _common.pick_device()
    pipe = pipeline("depth-estimation", model=MODEL_ID, device=device)

    predictions, started, no_reference = [], time.time(), 0
    for sample in samples:
        image = Image.open(sample["image"]).convert("RGB")
        relative = np.asarray(pipe(image)["predicted_depth"], dtype=np.float64)
        scale = _common.reference_scale(relative, sample, inverse=INVERSE)
        if not np.isfinite(scale):
            no_reference += 1
        for obj in sample["objects"]:
            value = _common.sample_depth(
                relative, obj["bbox"], (sample["width"], sample["height"])
            )
            predictions.append({
                "obj_id": obj["obj_id"], "image": sample["image"],
                "pred_depth_m": _common.apply_scale(value, scale, inverse=INVERSE),
            })
    _common.write_result(
        args.out, "Depth Anything V2", "relative + reference rescale", predictions,
        (time.time() - started) / max(len(samples), 1), device,
        notes=(
            f"Inverse-depth assumed; rescaled per image from one known-size object. "
            f"{no_reference} of {len(samples)} images had no usable reference."
        ),
    )


if __name__ == "__main__":
    main()
