"""One row per physical car, not one per detection.

A parked car seen in forty frames yields forty detections. Collapsing them is the
part of this that is genuinely hard, and the part most demos skip by reporting
detections and calling them objects.

Association happens in **ground coordinates**, not the image. A static car
projects to the same ground point from every viewpoint, so camera motion is
simply irrelevant and clustering is nearest-neighbour in metres. That is not a
stylistic preference: measured on this footage, a fixed ground point moves tens
of pixels between consecutive frames against a car box of similar size, so
inter-frame IoU is near zero and image-plane association cannot work at all.

Two failure modes pull against each other:

* **over-merge** -- bays sit about 2.5 m apart and a fix carries ~0.5 m of sigma,
  so a generous radius fuses neighbouring cars into one row and the count comes
  out low. This is the dangerous direction: the output looks cleaner.
* **under-merge** -- one car whose fixes scatter across the pass becomes two
  rows, and the count comes out high.

Both are detectable without ground truth, because a cluster's own scatter is a
measurement. If the spread of a cluster's fixes far exceeds what the ranging
predicted, the cluster is holding more than one car. `audit()` reports that, and
it is the number to look at before believing any count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MERGE_RADIUS_M = 2.5
"""How close two ground fixes must be to be the same car.

Roughly one car width, and deliberately below the ~2.5-3 m spacing of adjacent
bays so the default errs toward splitting rather than fusing. Splitting is
recoverable by inspection; fusing hides a car."""

MIN_FIXES = 3
"""Fewest fixes before a cluster is reported as a car.

A single detection that never recurs is far more likely to be a false positive
than a car seen once, and two points cannot show a scatter."""


@dataclass
class Cluster:
    """Fixes believed to belong to one physical car."""

    cluster_id: int
    points: list = field(default_factory=list)
    sigmas: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    classes: list = field(default_factory=list)

    @property
    def centre(self) -> np.ndarray:
        return np.mean(self.points, axis=0)

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def spread_m(self) -> float:
        """RMS distance of the fixes from their own centre."""
        if self.n < 2:
            return 0.0
        delta = np.asarray(self.points) - self.centre
        return float(np.sqrt(np.mean(np.sum(delta**2, axis=1))))

    @property
    def predicted_sigma_m(self) -> float:
        """What the ranging said a single fix of this car should scatter by."""
        return float(np.median(self.sigmas)) if self.sigmas else float("nan")

    @property
    def consistency(self) -> float:
        """Observed spread over predicted. Near 1 means the error bars are honest.

        Well above 1 means this cluster is probably two cars, or the ranging is
        optimistic. Either way it should not be reported as a confident position.
        """
        predicted = self.predicted_sigma_m
        return self.spread_m / predicted if predicted > 0 else float("nan")

    @property
    def position_sigma_m(self) -> float:
        """Uncertainty of the *mean*, which is what gets published."""
        return self.spread_m / np.sqrt(self.n) if self.n else float("inf")

    @property
    def top_class(self) -> str:
        return max(set(self.classes), key=self.classes.count) if self.classes else "unknown"

    @property
    def confidence(self) -> float:
        return float(np.median(self.scores)) if self.scores else 0.0


def cluster_fixes(records: list[dict], radius_m: float = MERGE_RADIUS_M) -> list[Cluster]:
    """Group ground fixes into physical cars.

    Single-linkage against a fixed radius, with the cluster centre updated as it
    grows. Deliberately the simplest thing that can work: any weakness in the
    result should be attributable to the *frame* the association happens in
    rather than to a clever algorithm hiding the geometry.
    """
    clusters: list[Cluster] = []
    centres: list[np.ndarray] = []

    for record in records:
        point = np.asarray(record["xy"], dtype=float)
        best, best_distance = None, radius_m
        for index, centre in enumerate(centres):
            distance = float(np.linalg.norm(centre - point))
            if distance < best_distance:
                best, best_distance = index, distance
        if best is None:
            best = len(clusters)
            clusters.append(Cluster(cluster_id=best))
            centres.append(point.copy())
        cluster = clusters[best]
        cluster.points.append(point)
        cluster.sigmas.append(record["sigma"])
        cluster.frames.append(record["frame_id"])
        cluster.scores.append(record.get("score", 1.0))
        cluster.classes.append(record.get("cls", "car"))
        centres[best] = cluster.centre
    return clusters


def audit(clusters: list[Cluster], radius_m: float = MERGE_RADIUS_M) -> dict:
    """Evidence about whether the merge radius was right, without ground truth.

    Three signals, none of which needs a label:

    ``suspect``      clusters whose scatter far exceeds the predicted sigma --
                     candidates for holding two cars.
    ``near_pairs``   distinct clusters closer together than the radius. These
                     survived only because clustering is order-dependent, and
                     each is a coin-flip that could have gone the other way.
    ``singletons``   clusters below the fix threshold, dropped from the output.
                     A large count means detection is unstable, not that the
                     scene has many cars.
    """
    kept = [c for c in clusters if c.n >= MIN_FIXES]
    dropped = [c for c in clusters if c.n < MIN_FIXES]
    suspect = [c for c in kept if np.isfinite(c.consistency) and c.consistency > 2.0]

    near = 0
    if len(kept) > 1:
        centres = np.array([c.centre for c in kept])
        distances = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        near = int((distances.min(axis=1) < radius_m).sum() // 2)

    consistencies = [c.consistency for c in kept if np.isfinite(c.consistency)]
    return {
        "clusters": len(clusters),
        "reported": len(kept),
        "singletons": len(dropped),
        "suspect": len(suspect),
        "near_pairs": near,
        "median_consistency": float(np.median(consistencies)) if consistencies else float("nan"),
        "median_fixes": float(np.median([c.n for c in kept])) if kept else float("nan"),
    }


def sweep(records: list[dict], radii=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)) -> list[dict]:
    """Car count against merge radius.

    The honest way to present a count that depends on a threshold. A genuine
    population shows a plateau -- a range of radii over which the count barely
    moves, because real cars are separated by more than the fix error. No
    plateau means the threshold is doing the deciding, and the count is an
    artefact of it.
    """
    out = []
    for radius in radii:
        clusters = cluster_fixes(records, radius_m=radius)
        summary = audit(clusters, radius_m=radius)
        summary["radius_m"] = radius
        out.append(summary)
    return out
