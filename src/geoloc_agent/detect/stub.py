"""StubDetector: replays scripted detections from JSON or an in-memory table.

No ML anywhere in the loop. This is what lets the whole pipeline be tested
end-to-end on day one, and what keeps the eval harness honest later -- with a
perfect detector in the loop, any error you measure is geometry and filter
error, not detector error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geoloc_agent.contracts import Detection, Frame
from geoloc_agent.detect.base import Detector


class StubDetector(Detector):
    name = "stub"

    def __init__(self, table: dict[int, list[Detection]] | None = None) -> None:
        self.table = table or {}

    @classmethod
    def from_json(cls, path: str | Path) -> StubDetector:
        payload = json.loads(Path(path).read_text())
        table: dict[int, list[Detection]] = {}
        for fid, dets in payload.get("frames", {}).items():
            frame_id = int(fid)
            table[frame_id] = [
                Detection(
                    bbox=np.asarray(d["bbox"], dtype=float),
                    cls=d.get("cls", "unknown"),
                    score=float(d.get("score", 1.0)),
                    frame_id=frame_id,
                    track_hint=d.get("track_hint"),
                )
                for d in dets
            ]
        return cls(table)

    @classmethod
    def from_session(cls, session) -> StubDetector:
        table = session.scripted_detections()
        if table is None:
            raise ValueError(f"{type(session).__name__} has no scripted detections")
        return cls(table)

    def detect(self, frame: Frame) -> list[Detection]:
        return list(self.table.get(frame.frame_id, []))
