"""Roboflow-hosted detection, for imagery COCO models cannot read.

Straight-down satellite defeats a COCO-trained detector. Measured on the exact
tiles this demo uses, YOLO11n returns `clock`, `train`, `potted plant` and
`person`, finding two cars at the loosest threshold it has. It is not a
confidence problem: a car photographed from directly above is a rectangle, and
the model has never seen one.

Roboflow Universe hosts models trained on overhead imagery, which is the gap.
This wraps their hosted inference so it satisfies the same one-method contract
the rest of the pipeline uses -- ``detect(image) -> [Detection]`` -- and drops in
wherever the local CoreML detector goes.

Tiling still matters and for the same reason it did on the drone footage: at
zoom 20 a car is about 33 px, and handing a 1024 px mosaic to a 640 px network
shrinks it below what any detector can hold. Tiles are sent at native resolution
so the car arrives at the size the model was trained to see.

Needs `ROBOFLOW_API_KEY` in the environment or `.env`. The key is read at call
time and never logged.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np

from carloc.detect import NMS_IOU, Detection, _suppress

DEFAULT_MODEL = "aerial-vehicle-detection/1"
"""Overridden per deployment. Any Universe model whose classes include vehicles
works; the wrapper maps whatever labels it returns onto `VEHICLE_LABELS`."""

VEHICLE_LABELS = {
    "car", "cars", "vehicle", "vehicles", "truck", "bus", "van",
    "pickup", "small-vehicle", "large-vehicle", "motorbike", "motorcycle",
}
"""Universe models disagree wildly about class names, so match generously and
case-insensitively rather than assuming one taxonomy."""

TILE = 320
OVERLAP = 0.25


def api_key() -> str:
    """Read the key from the environment, falling back to a local .env.

    Deliberately not cached and never printed: it is a credential, and the rest
    of this module handles it only long enough to pass it to the client.
    """
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key.strip()
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ROBOFLOW_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "no ROBOFLOW_API_KEY found. Add it to .env as ROBOFLOW_API_KEY=... "
        "(free key at roboflow.com)"
    )


def _encode(image: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode()


class RoboflowDetector:
    """Hosted Roboflow model, tiled, behind the pipeline's detector contract."""

    def __init__(self, model_id: str = DEFAULT_MODEL, confidence: float = 0.25,
                 tile: int = TILE, overlap: float = OVERLAP,
                 api_url: str = "https://detect.roboflow.com") -> None:
        self.model_id = model_id
        self.confidence = confidence
        self.tile = tile
        self.overlap = overlap
        self.api_url = api_url
        self._client = None
        self.last_labels: dict[str, int] = {}
        """Every class name the model returned last call, vehicle or not.

        Kept because the first question when a Universe model finds nothing is
        whether it found *something else* -- which is how the COCO failure was
        diagnosed, and the same trap applies here."""

    def load(self) -> None:
        if self._client is not None:
            return
        from inference_sdk import InferenceHTTPClient

        self._client = InferenceHTTPClient(api_url=self.api_url, api_key=api_key())

    def tiles_for(self, width: int, height: int):
        size = min(self.tile, width, height)
        step = max(1, int(size * (1.0 - self.overlap)))
        xs = list(range(0, max(width - size, 0) + 1, step))
        ys = list(range(0, max(height - size, 0) + 1, step))
        if xs[-1] + size < width:
            xs.append(width - size)
        if ys[-1] + size < height:
            ys.append(height - size)
        return [(x, y, size, size) for y in ys for x in xs]

    def _infer(self, crop: np.ndarray, offset_x: int, offset_y: int) -> list[Detection]:
        result = self._client.infer(_encode(crop), model_id=self.model_id)
        found: list[Detection] = []
        for prediction in (result or {}).get("predictions", []):
            label = str(prediction.get("class", "")).lower()
            self.last_labels[label] = self.last_labels.get(label, 0) + 1
            score = float(prediction.get("confidence", 0.0))
            if score < self.confidence or label not in VEHICLE_LABELS:
                continue
            # Roboflow returns centre-x/centre-y plus width/height in pixels.
            cx = float(prediction["x"]) + offset_x
            cy = float(prediction["y"]) + offset_y
            w = float(prediction["width"])
            h = float(prediction["height"])
            if w < 2 or h < 2:
                continue
            found.append(Detection(
                bbox=np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]),
                cls="car", score=score))
        return found

    def detect(self, image: np.ndarray) -> list[Detection]:
        self.load()
        self.last_labels = {}
        height, width = image.shape[:2]
        found: list[Detection] = []
        for x, y, tw, th in self.tiles_for(width, height):
            found.extend(self._infer(image[y:y + th, x:x + tw], x, y))
        return _suppress(found, NMS_IOU)


def find_models(query: str = "aerial vehicle", limit: int = 12) -> list[dict]:
    """Search Universe for a candidate model.

    Returns id, name and class list so a model can be picked on what it actually
    detects rather than on its title.
    """
    import json
    import urllib.parse
    import urllib.request

    url = ("https://api.roboflow.com/search?"
           + urllib.parse.urlencode({"query": query, "limit": limit,
                                     "api_key": api_key()}))
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode())
    out = []
    for item in payload.get("results", payload.get("models", []))[:limit]:
        out.append({
            "id": item.get("id") or f"{item.get('workspace')}/{item.get('project')}",
            "name": item.get("name") or item.get("project"),
            "classes": item.get("classes") or item.get("class_names") or [],
            "images": item.get("images"),
        })
    return out
