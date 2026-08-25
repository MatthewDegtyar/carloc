"""Depth Anything V2, metric outdoor checkpoint. Output treated as metres."""

import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"


def main():
    args = _common.parse_args()
    manifest = _common.load_manifest(args.manifest)
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]

    from transformers import pipeline

    device = args.device or _common.pick_device()
    pipe = pipeline("depth-estimation", model=MODEL_ID, device=device)

    predictions, started = [], time.time()
    for sample in samples:
        image = Image.open(sample["image"]).convert("RGB")
        depth = np.asarray(pipe(image)["predicted_depth"], dtype=np.float64)
        for obj in sample["objects"]:
            predictions.append({
                "obj_id": obj["obj_id"], "image": sample["image"],
                "pred_depth_m": _common.sample_depth(
                    depth, obj["bbox"], (sample["width"], sample["height"])
                ),
            })
    _common.write_result(
        args.out, "Depth Anything V2", "metric-outdoor-large", predictions,
        (time.time() - started) / max(len(samples), 1), device,
        notes="Metric checkpoint, output used as metres with no rescaling.",
    )


if __name__ == "__main__":
    main()
