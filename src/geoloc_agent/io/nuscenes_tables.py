"""Minimal nuScenes metadata reader.

A drop-in for the two devkit calls this project needs -- ``.scene`` and
``.get(table, token)`` -- built straight from the JSON tables.

Why not just use ``nuscenes-devkit``: it pulls shapely (needs system GEOS),
opencv, scikit-learn and matplotlib, which is a large and brittle dependency
chain for what amounts to reading nine JSON files and doing three joins. The
devkit remains supported -- ``NuScenesSession`` takes an injected ``nusc``
object, so either works -- but this keeps the loader importable in CI and on a
bare Python.

Three fields the devkit *derives* rather than stores, reproduced here because
downstream code expects them:

* ``sample['data']``   channel -> sample_data token, keyframes only
* ``sample['anns']``   the annotation tokens for that sample
* ``sample_annotation['category_name']`` joined through instance -> category
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

TABLES = (
    "log", "scene", "sample", "sample_data", "ego_pose", "calibrated_sensor",
    "sensor", "sample_annotation", "instance", "category", "visibility",
)


class NuScenesTables:
    """Reads the metadata JSON and answers the same questions the devkit does."""

    def __init__(self, dataroot: str | Path, version: str = "v1.0-mini") -> None:
        self.dataroot = Path(dataroot)
        self.version = version
        root = self.dataroot / version
        if not root.is_dir():
            raise FileNotFoundError(
                f"no nuScenes metadata at {root}. Expected the extracted archive, i.e. "
                f"{self.dataroot}/{version}/sample.json"
            )

        self._tables: dict[str, list[dict]] = {}
        for name in TABLES:
            path = root / f"{name}.json"
            self._tables[name] = json.loads(path.read_text()) if path.exists() else []

        self._index: dict[str, dict[str, dict]] = {
            name: {record["token"]: record for record in records}
            for name, records in self._tables.items()
        }
        self._derive()

    # -- devkit-compatible surface ---------------------------------------

    @property
    def scene(self) -> list[dict]:
        return self._tables["scene"]

    @property
    def sample(self) -> list[dict]:
        return self._tables["sample"]

    def get(self, table: str, token: str) -> dict:
        try:
            return self._index[table][token]
        except KeyError as exc:
            raise KeyError(f"no record {token!r} in table {table!r}") from exc

    def __len__(self) -> int:
        return len(self._tables["sample"])

    # -- derived fields ---------------------------------------------------

    def _derive(self) -> None:
        # sample_data channel comes via calibrated_sensor -> sensor.
        channel_of: dict[str, str] = {}
        for calibrated in self._tables["calibrated_sensor"]:
            sensor = self._index["sensor"].get(calibrated["sensor_token"])
            if sensor:
                channel_of[calibrated["token"]] = sensor["channel"]

        data_by_sample: dict[str, dict[str, str]] = defaultdict(dict)
        for record in self._tables["sample_data"]:
            record["channel"] = channel_of.get(record["calibrated_sensor_token"], "")
            if record.get("is_key_frame") and record["channel"]:
                data_by_sample[record["sample_token"]][record["channel"]] = record["token"]

        anns_by_sample: dict[str, list[str]] = defaultdict(list)
        for annotation in self._tables["sample_annotation"]:
            anns_by_sample[annotation["sample_token"]].append(annotation["token"])
            instance = self._index["instance"].get(annotation["instance_token"])
            category = (
                self._index["category"].get(instance["category_token"]) if instance else None
            )
            annotation["category_name"] = category["name"] if category else "unknown"

        for sample in self._tables["sample"]:
            sample["data"] = dict(data_by_sample.get(sample["token"], {}))
            sample["anns"] = list(anns_by_sample.get(sample["token"], []))
