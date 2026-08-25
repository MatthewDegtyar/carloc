"""Session loader interface.

Every data source -- synthetic, nuScenes, Stray Scanner, bare video -- implements
this and nothing else. Downstream code never learns which one it is holding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from geoloc_agent.contracts import Detection, Frame
from geoloc_agent.geo import GeoOrigin


@dataclass
class TruthObject:
    """Ground-truth object position in the world (local ENU) frame.

    ``positions`` carries per-frame truth for objects that move; ``position`` is
    the static case and the fallback.
    """

    obj_id: str
    position: np.ndarray
    cls: str = "unknown"
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    positions: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)

    def at(self, frame_id: int) -> np.ndarray:
        return self.positions.get(frame_id, self.position)


class Session(ABC):
    """A posed image sequence plus, in eval mode, its ground truth."""

    name: str = "session"

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Yield frames in time order. May be a generator over lazily loaded images."""

    @abstractmethod
    def truth(self) -> dict[str, TruthObject]:
        """Ground-truth objects, keyed by id. Empty dict when unavailable."""

    @property
    def origin(self) -> GeoOrigin | None:
        """Origin of the local ENU frame, or None if the session is metric-only."""
        return None

    def scripted_detections(self) -> dict[int, list[Detection]] | None:
        """Detections known without running a detector (synthetic or replayed).

        Returns None for sessions that require real perception.
        """
        return None

    def truth_at(self, frame_id: int) -> dict[str, np.ndarray]:
        return {oid: obj.at(frame_id) for oid, obj in self.truth().items()}

    def __len__(self) -> int:
        return sum(1 for _ in self.frames())
