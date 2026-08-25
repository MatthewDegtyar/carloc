"""Detector interface.

Perception is the fast loop (milliseconds). It sees a Frame and returns
Detections. It knows nothing about world geometry, ranging, or tracks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from geoloc_agent.contracts import Detection, Frame


class Detector(ABC):
    name: str = "detector"

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        """Detections for this frame, in image coordinates."""

    def warmup(self) -> None:
        """Optional: pay one-time model load cost before benchmarking."""
        return None
