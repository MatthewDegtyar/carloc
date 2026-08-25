"""Report generation for depthbench."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from depthbench.metrics import ModelScore, saturation_span
from depthbench.schema import Manifest


def _f(value, spec=".2f", dash="--"):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return dash
    return format(value, spec)


def build_report(
    scores: list[ModelScore], manifest: Manifest, out: Path, depth_key: str = "surface_depth_m"
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = [s for s in scores if not s.failed]
    ok.sort(key=lambda s: (np.isnan(s.median_abs_err), s.median_abs_err))
    failed = [s for s in scores if s.failed]

    depths = [getattr(o, depth_key) for s in manifest.samples for o in s.objects]
    lines: list[str] = []
    add = lines.append

    add("# Monocular depth benchmark, 0-50 m")
    add("")
    add(
        f"{len(manifest.samples)} images, {len(depths)} objects, "
        f"{min(depths):.1f}-{max(depths):.1f} m. Ground truth: {manifest.source}, "
        f"real intrinsics, 3-D box annotations."
    )
    add("")
    add(
        f"Scored against **{depth_key.replace('_', ' ')}**. Depth models predict the "
        f"distance to the visible *surface*; a geolocation pipeline wants the object "
        f"*centroid*. Those differ by about half an object's length -- a median of "
        f"1.0 m here -- which is comparable to the errors being measured, so the "
        f"benchmark computes both and neither is charged to a model silently. "
        f"Re-run with `--depth centroid` for the other view."
    )
    add("")

    add("## Results")
    add("")
    add(
        "| model | n | coverage | median abs err | p90 abs err | median rel err | "
        "delta<1.25 | rank corr | s/image |"
    )
    add("|" + "---|" * 9)
    for s in ok:
        add(
            f"| {s.label} | {s.n} | {s.coverage * 100:.0f}% | **{_f(s.median_abs_err)} m** | "
            f"{_f(s.p90_abs_err)} m | {_f(s.median_rel_err * 100, '.0f')}% | "
            f"{_f(s.delta1 * 100, '.0f')}% | {_f(s.spearman, '.2f')} | "
            f"{_f(s.seconds_per_image)} |"
        )
    add("")
    add(
        "`delta<1.25` is the standard depth metric: the fraction of objects whose "
        "predicted and true depth are within 25% of each other. `rank corr` is "
        "Spearman against truth -- if it is near zero the metric error is noise, and "
        "a negative value means the depth map was read upside down."
    )
    add("")

    add("## Error by range")
    add("")
    add("Median absolute error in metres:")
    add("")
    header = "| model | " + " | ".join(f"{lo}-{hi} m" for lo, hi in
                                       [(b.lo, b.hi) for b in (ok[0].buckets if ok else [])])
    add(header + " |")
    add("|" + "---|" * (len(ok[0].buckets) + 1 if ok else 2))
    for s in ok:
        cells = " | ".join(f"{_f(b.median_abs_err)}" for b in s.buckets)
        add(f"| {s.label} | {cells} |")
    add("")

    add("## Saturation")
    add("")
    add(
        "The characteristic failure of monocular depth: predictions compress into a "
        "narrow band, so the model cannot distinguish near from far even though its "
        "average error looks tolerable. `span ratio` is the predicted spread across "
        "buckets divided by the true spread -- 1.0 tracks range correctly, near 0 "
        "means the answer barely changes with distance."
    )
    add("")
    add("| model | median pred @0-10 m | @40-50 m | span ratio |")
    add("|---|---|---|---|")
    for s in ok:
        if len(s.buckets) < 2:
            continue
        add(
            f"| {s.label} | {_f(s.buckets[0].median_pred)} m | "
            f"{_f(s.buckets[-1].median_pred)} m | **{_f(saturation_span(s))}** |"
        )
    add("")

    add("## Speed")
    add("")
    add("| model | s/image | device |")
    add("|---|---|---|")
    for s in sorted(ok, key=lambda s: (np.isnan(s.seconds_per_image), s.seconds_per_image)):
        add(f"| {s.label} | {_f(s.seconds_per_image)} | {s.device} |")
    add("")

    if failed:
        add("## Did not run")
        add("")
        for s in failed:
            add(f"- **{s.label}** — {s.error.strip().splitlines()[-1][:220] if s.error else '?'}")
        add("")

    add("## What this says about picking one")
    add("")
    add(
        "The overall median hides the thing that decides usability, which is where "
        "the model stops tracking range at all. Read the saturation table with the "
        "per-bucket table: a model can post a good average by being excellent up "
        "close and simply returning a constant beyond 30 m."
    )
    add("")

    add("## Method")
    add("")
    add(
        "- Each model runs in its own virtualenv as a subprocess, talking to the "
        "harness over JSON. They disagree about torch, timm, transformers and numpy "
        "versions; one shared environment silently downgrades something and then you "
        "are benchmarking the downgrade."
    )
    add(
        "- Depth is sampled as the median over the central 50% of each ground-truth "
        "box. A whole box contains background through windows and around outlines, "
        "which sits tens of metres further away."
    )
    add(
        "- Sampling is identical for every model, in one shared helper, so the "
        "comparison is between models rather than between sampling choices."
    )
    add(
        "- Objects are filtered to >=60% visibility, fully in front of the camera, "
        "at least 24 px, and inside 50 m."
    )
    add(
        "- Predictions that come back non-finite are recorded as misses, not dropped, "
        "so a model cannot look accurate by declining the hard objects. That is what "
        "`coverage` reports."
    )
    add("")
    for s in scores:
        if s.notes:
            add(f"- **{s.label}**: {s.notes}")
    add("")

    out.write_text("\n".join(lines))
    return out
