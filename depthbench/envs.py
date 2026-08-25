"""One virtualenv per model.

These models disagree about torch, timm, transformers and numpy versions. A single
environment holding all of them either fails to resolve or silently downgrades
something, and you are then benchmarking the downgrade rather than the model.

`numpy<2` appears repeatedly: several of these stacks call `int()` or `float()` on
size-1 arrays, which numpy 2 rejects. It is a real constraint, not caution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ENV_ROOT = "/tmp/depthbench-envs"


@dataclass(frozen=True)
class ModelEnv:
    name: str
    runner: str
    packages: list[str] = field(default_factory=list)
    python: str = "3.12"
    variant: str = ""
    post_install: list[list[str]] = field(default_factory=list)
    notes: str = ""

    @property
    def venv(self) -> str:
        return f"{ENV_ROOT}/{self.name}/.venv"


TORCH = ["torch", "torchvision", "pillow", "numpy<2"]

MODELS: dict[str, ModelEnv] = {
    "depth_pro": ModelEnv(
        name="depth_pro",
        runner="depth_pro_runner.py",
        packages=[*TORCH, "git+https://github.com/apple/ml-depth-pro", "transformers"],
        notes="Estimates its own focal length; no intrinsics supplied.",
    ),
    "metric3d_real_K": ModelEnv(
        name="metric3d_real_K",
        runner="metric3d_runner.py",
        variant="real intrinsics",
        packages=[*TORCH, "timm", "mmengine", "scipy"],
        post_install=[["__install_mmcv_shim__"]],
        notes="Given the true camera intrinsics from the manifest. Uses an mmcv shim.",
    ),
    "metric3d_default_K": ModelEnv(
        name="metric3d_default_K",
        runner="metric3d_runner.py",
        variant="default intrinsics",
        packages=[*TORCH, "timm", "mmengine", "scipy"],
        post_install=[["__install_mmcv_shim__"]],
        notes="Given the model's own default intrinsics, to isolate their effect.",
    ),
    "dav2_metric_outdoor": ModelEnv(
        name="dav2_metric_outdoor",
        runner="dav2_metric_runner.py",
        packages=[*TORCH, "transformers", "accelerate"],
        notes="Metric checkpoint; output taken as metres directly.",
    ),
    "dav2_relative_rescaled": ModelEnv(
        name="dav2_relative_rescaled",
        runner="dav2_relative_runner.py",
        packages=[*TORCH, "transformers", "accelerate"],
        notes="Relative map rescaled per image from one known-size object.",
    ),
    "yolo26n_depth": ModelEnv(
        name="yolo26n_depth",
        runner="yolo_depth_runner.py",
        variant="yolo26n-depth.pt",
        packages=["ultralytics", "numpy<2"],
        notes="Speed baseline.",
    ),
    "yolo26x_depth": ModelEnv(
        name="yolo26x_depth",
        runner="yolo_depth_runner.py",
        variant="yolo26x-depth.pt",
        packages=["ultralytics", "numpy<2"],
        notes="Speed baseline, large.",
    ),
}
