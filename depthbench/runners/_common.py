"""Helpers shared by every runner.

Imported by path, not as a package: runners execute inside foreign virtualenvs
that cannot import `depthbench`. numpy is the only dependency, and every model
environment has it.

Keeping the sampling here rather than per-runner is the point. If each model
sampled its own depth map differently, the benchmark would be comparing sampling
choices as much as models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CENTRAL_FRACTION = 0.5
"""Fraction of the box used for sampling.

A bounding box contains background -- through windows, under a chassis, around a
pedestrian's outline -- and those pixels belong to whatever is behind the object,
often tens of metres further away. Taking the median of the central portion keeps
the sample on the object. Using the whole box would penalise every model equally
but for a reason that has nothing to do with depth estimation."""


def load_manifest(path):
    return json.loads(Path(path).read_text())


def sample_depth(depth_map: np.ndarray, bbox, image_wh) -> float:
    """Median depth over the central region of a box, in the map's own units.

    The map is resampled by index rather than interpolated: models return depth at
    their own working resolution, and interpolating a depth map across an object
    boundary invents values that lie between foreground and background.
    """
    depth_map = np.asarray(depth_map, dtype=np.float64)
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze()
    if depth_map.ndim != 2:
        return float("nan")

    width, height = image_wh
    map_h, map_w = depth_map.shape
    sx, sy = map_w / float(width), map_h / float(height)

    x1, y1, x2, y2 = (float(v) for v in bbox)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    hw = 0.5 * CENTRAL_FRACTION * (x2 - x1)
    hh = 0.5 * CENTRAL_FRACTION * (y2 - y1)

    a = max(int(round((cx - hw) * sx)), 0)
    b = min(int(round((cx + hw) * sx)) + 1, map_w)
    c = max(int(round((cy - hh) * sy)), 0)
    d = min(int(round((cy + hh) * sy)) + 1, map_h)
    if b <= a or d <= c:
        return float("nan")

    patch = depth_map[c:d, a:b]
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return float("nan")
    return float(np.median(patch))


def reference_scale(depth_map, sample, inverse: bool) -> float:
    """Scale factor turning a relative map into metres, from one known-size object.

    Uses the pinhole relation ``Z = fy * H_real / h_px`` on the reference box to
    get its metric depth, then the ratio to the model's relative value there.

    The reference depth this yields is biased: a 2-D box is the convex hull of a
    3-D object and stands taller than the object itself, which makes ``h_px`` too
    large and the implied depth too short -- about 20% for a car. That bias is not
    an artefact of the harness, it is what this rescaling method actually costs,
    and it is inherited by every estimate in the image. Reported, not corrected.
    """
    reference = sample.get("reference")
    if not reference:
        return float("nan")
    x1, y1, x2, y2 = reference["bbox"]
    h_px = float(y2 - y1)
    if h_px <= 1:
        return float("nan")
    fy = float(sample["K"][1][1])
    metric_depth = fy * float(reference["height_m"]) / h_px

    relative = sample_depth(depth_map, reference["bbox"], (sample["width"], sample["height"]))
    if not np.isfinite(relative) or relative <= 0:
        return float("nan")
    # Inverse-depth (disparity) models: metric = scale / value. Direct models:
    # metric = scale * value.
    return metric_depth * relative if inverse else metric_depth / relative


def apply_scale(value: float, scale: float, inverse: bool) -> float:
    if not (np.isfinite(value) and np.isfinite(scale)) or value <= 0:
        return float("nan")
    return scale / value if inverse else scale * value


def write_result(path, model, variant, predictions, seconds_per_image, device, notes="",
                 failed=False, error=""):
    payload = {
        "model": model, "variant": variant, "predictions": predictions,
        "seconds_per_image": seconds_per_image, "device": device, "notes": notes,
        "failed": failed, "error": error,
    }
    # Write-then-rename, so an interrupted or failed run cannot destroy the
    # previous good result. A killed Depth Pro run wiped a complete 74-image
    # result this way, and the only recovery was to run it again.
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(target)
    print(f"wrote {path}: {len(predictions)} predictions, {seconds_per_image:.2f}s/image")


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="")
    return parser.parse_args(argv if argv is not None else sys.argv[1:])
