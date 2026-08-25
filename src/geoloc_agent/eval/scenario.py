"""Scenario and sweep specs. YAML in, runs out -- no Python edits to add a case."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from geoloc_agent.fuse.tracker import TrackerConfig
from geoloc_agent.io.synthetic import SyntheticScenario, SyntheticSession
from geoloc_agent.noise import NoiseModel


@dataclass
class Scenario:
    """One evaluable configuration: a session, a noise model, and filter settings."""

    name: str
    source: str = "synthetic"
    seeds: int = 10
    bearing_sigma_px: float = 2.0
    use_ranger: bool = False
    range_every_n: int = 3
    synthetic: dict = field(default_factory=dict)
    noise: dict = field(default_factory=dict)
    tracker: dict = field(default_factory=dict)
    nuscenes: dict = field(default_factory=dict)
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> Scenario:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> list[Scenario]:
        payload = yaml.safe_load(Path(path).read_text())
        entries = payload["scenarios"] if isinstance(payload, dict) else payload
        return [cls.from_dict(entry) for entry in entries]

    def noise_model(self, overrides: dict | None = None) -> NoiseModel:
        merged = {**self.noise, **(overrides or {})}
        return NoiseModel.from_dict(merged)

    def tracker_config(self) -> TrackerConfig:
        known = {f for f in TrackerConfig.__dataclass_fields__}
        unknown = set(self.tracker) - known
        if unknown:
            raise ValueError(f"unknown tracker keys: {sorted(unknown)}")
        return TrackerConfig(**self.tracker)

    def build_session(self):
        if self.source == "synthetic":
            spec = SyntheticScenario.from_dict({"name": self.name, **self.synthetic})
            return SyntheticSession(spec)
        if self.source == "nuscenes":
            from geoloc_agent.io.nuscenes import NuScenesSession

            return NuScenesSession(**self.nuscenes)
        raise ValueError(f"unknown session source: {self.source}")


@dataclass
class Sweep:
    """A grid over noise parameters.

    One axis varied at a time by default. Full factorial grids over six noise
    knobs produce thousands of runs and a plot nobody reads; the point of a sweep
    is an error-vs-noise curve per parameter, which is what ``axes`` gives.
    """

    name: str
    scenario: str
    axes: dict[str, list[float]] = field(default_factory=dict)
    grid: dict[str, list[float]] = field(default_factory=dict)
    seeds: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> list[Sweep]:
        payload = yaml.safe_load(Path(path).read_text())
        entries = payload["sweeps"] if isinstance(payload, dict) else payload
        return [cls(**entry) for entry in entries]

    def points(self) -> list[tuple[str, dict[str, Any]]]:
        """(label, noise-override) pairs covering the sweep."""
        out: list[tuple[str, dict[str, Any]]] = []
        for parameter, values in self.axes.items():
            for value in values:
                out.append((f"{parameter}={value}", {parameter: value}))
        if self.grid:
            names = list(self.grid)
            for combo in product(*(self.grid[n] for n in names)):
                override = dict(zip(names, combo, strict=True))
                label = ", ".join(f"{k}={v}" for k, v in override.items())
                out.append((label, override))
        return out
