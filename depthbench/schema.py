"""The file contract between the scorer and the model runners.

Deliberately plain JSON. Runners execute in foreign virtualenvs that cannot import
this package, so the contract has to be readable without it -- these dataclasses
document and validate the shape, they are not a shared dependency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_VERSION = 1


@dataclass
class RefObject:
    """An object with known real-world size, for rescaling a relative depth map."""

    bbox: list[float]
    """(x1, y1, x2, y2) in original image pixels."""

    height_m: float
    """True world height of the object this box encloses."""

    cls: str = ""


@dataclass
class GtObject:
    obj_id: str
    bbox: list[float]
    cls: str
    surface_depth_m: float
    """Depth to the near face of the 3-D box: what a depth model predicts."""

    centroid_depth_m: float
    """Depth to the object centre: what a geolocation pipeline needs."""

    height_m: float = 0.0
    visibility: int = 4


@dataclass
class Sample:
    image: str
    width: int
    height: int
    K: list[list[float]]
    objects: list[GtObject] = field(default_factory=list)
    reference: RefObject | None = None
    """Known-size object used to rescale relative-depth models. Chosen per image
    rather than per object so a relative model is scaled once, as it would be in
    practice."""


@dataclass
class Manifest:
    version: int = MANIFEST_VERSION
    source: str = ""
    max_range_m: float = 50.0
    samples: list[Sample] = field(default_factory=list)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @classmethod
    def read(cls, path: str | Path) -> Manifest:
        raw = json.loads(Path(path).read_text())
        if raw.get("version") != MANIFEST_VERSION:
            raise ValueError(f"manifest version {raw.get('version')} != {MANIFEST_VERSION}")
        samples = []
        for s in raw["samples"]:
            reference = RefObject(**s["reference"]) if s.get("reference") else None
            samples.append(
                Sample(
                    image=s["image"], width=s["width"], height=s["height"], K=s["K"],
                    objects=[GtObject(**o) for o in s["objects"]], reference=reference,
                )
            )
        return cls(version=raw["version"], source=raw.get("source", ""),
                   max_range_m=raw.get("max_range_m", 50.0), samples=samples)


@dataclass
class Prediction:
    obj_id: str
    image: str
    pred_depth_m: float
    """NaN means the model declined or produced nothing usable here. Recorded
    rather than dropped: a model that silently skips hard objects would otherwise
    score better than one that attempts them."""


@dataclass
class RunResult:
    model: str
    variant: str = ""
    predictions: list[Prediction] = field(default_factory=list)
    seconds_per_image: float = float("nan")
    device: str = ""
    notes: str = ""
    failed: bool = False
    error: str = ""

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @classmethod
    def read(cls, path: str | Path) -> RunResult:
        raw = json.loads(Path(path).read_text())
        return cls(
            model=raw["model"], variant=raw.get("variant", ""),
            predictions=[Prediction(**p) for p in raw.get("predictions", [])],
            seconds_per_image=raw.get("seconds_per_image", float("nan")),
            device=raw.get("device", ""), notes=raw.get("notes", ""),
            failed=raw.get("failed", False), error=raw.get("error", ""),
        )
