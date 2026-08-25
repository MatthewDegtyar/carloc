"""YOLO26 depth via ultralytics, as a speed baseline."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402


def extract_depth(result):
    """Pull a depth array out of an ultralytics Results object.

    The attribute name is not pinned across releases, so the candidates are tried
    in order and the first 2-D float array wins. If none matches, the runner fails
    loudly rather than reporting zeros.
    """
    for attr in ("depth", "depth_map", "predicted_depth", "proto"):
        value = getattr(result, attr, None)
        if value is None:
            continue
        data = getattr(value, "data", value)
        try:
            array = np.asarray(data.cpu().numpy() if hasattr(data, "cpu") else data)
        except Exception:  # noqa: BLE001
            continue
        array = array.squeeze()
        if array.ndim == 2 and np.issubdtype(array.dtype, np.floating):
            return array.astype(np.float64), attr
    return None, ""


def main():
    args = _common.parse_args()
    weights = os.environ.get("DEPTHBENCH_YOLO_WEIGHTS", "yolo26n-depth.pt")
    manifest = _common.load_manifest(args.manifest)
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]

    from ultralytics import YOLO

    model = YOLO(weights)
    device = args.device or "cpu"

    predictions, started, source_attr = [], time.time(), ""
    for sample in samples:
        result = model.predict(sample["image"], verbose=False, device=device)[0]
        depth, attr = extract_depth(result)
        source_attr = source_attr or attr
        if depth is None:
            _common.write_result(
                args.out, "YOLO26 depth", weights, [], float("nan"), device,
                failed=True,
                error=(
                    "no depth array on Results; attrs="
                    f"{[a for a in dir(result) if not a.startswith('_')][:25]}"
                ),
            )
            return
        for obj in sample["objects"]:
            predictions.append({
                "obj_id": obj["obj_id"], "image": sample["image"],
                "pred_depth_m": _common.sample_depth(
                    depth, obj["bbox"], (sample["width"], sample["height"])
                ),
            })
    _common.write_result(
        args.out, "YOLO26 depth", weights, predictions,
        (time.time() - started) / max(len(samples), 1), device,
        notes=f"depth read from Results.{source_attr}",
    )


if __name__ == "__main__":
    main()
