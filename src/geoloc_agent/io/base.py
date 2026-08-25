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
    rotation: np.ndarray | None = None
    """Body->world rotation. Real vehicles are not axis-aligned, and treating a
    rotated car as an axis-aligned box inflates its projected width by up to the
    ratio of its length to its width -- roughly 2.4x for a sedan."""
    rotations: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        if self.rotation is not None:
            self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)

    def at(self, frame_id: int) -> np.ndarray:
        return self.positions.get(frame_id, self.position)

    def rotation_at(self, frame_id: int) -> np.ndarray:
        rotation = self.rotations.get(frame_id, self.rotation)
        return np.eye(3) if rotation is None else rotation

    def corners(self, frame_id: int | None = None) -> np.ndarray:
        """The 8 corners of the oriented 3-D box, in world coordinates.

        nuScenes size order is (width, length, height) with the body frame's +x
        along width and +y along length.
        """
        w, length, h = self.size
        centre = self.position if frame_id is None else self.at(frame_id)
        R = np.eye(3) if frame_id is None else self.rotation_at(frame_id)
        if frame_id is None and self.rotation is not None:
            R = self.rotation
        signs = np.array(
            [
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            ],
            dtype=float,
        )
        local = signs * np.array([w / 2, length / 2, h / 2])
        return (R @ local.T).T + centre


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
