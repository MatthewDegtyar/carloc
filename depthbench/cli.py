"""depthbench: manifest -> setup -> run -> score."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from depthbench.envs import MODELS, ModelEnv
from depthbench.schema import Manifest, RunResult

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
DEFAULT_MANIFEST = DATA / "manifest.json"


def _uv(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["uv", *args], **kwargs)


def setup(env: ModelEnv, force: bool = False) -> bool:
    """Build this model's virtualenv. Returns False if it could not be built."""
    root = Path(env.venv).parent
    root.mkdir(parents=True, exist_ok=True)
    if force and Path(env.venv).exists():
        subprocess.run(["rm", "-rf", env.venv], check=False)

    if not Path(env.venv).exists():
        result = _uv("venv", "--python", env.python, env.venv, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[{env.name}] venv failed: {result.stderr[-400:]}")
            return False

    environ = {**os.environ, "VIRTUAL_ENV": env.venv}
    result = subprocess.run(
        ["uv", "pip", "install", "-q", *env.packages],
        env=environ, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[{env.name}] install failed:\n{result.stderr[-900:]}")
        return False
    for command in env.post_install:
        if command == ["__install_mmcv_shim__"]:
            _install_mmcv_shim(env)
            continue
        subprocess.run(command, env=environ, cwd=root, check=False)
    print(f"[{env.name}] ready")
    return True


def _install_mmcv_shim(env: ModelEnv) -> None:
    """Drop the mmcv stand-in into this venv. See depthbench/shims/mmcv."""
    import shutil

    site = next(Path(env.venv).glob("lib/python*/site-packages"), None)
    if site is None:
        print(f"[{env.name}] could not locate site-packages for the mmcv shim")
        return
    target = site / "mmcv"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ROOT / "shims" / "mmcv", target)
    print(f"[{env.name}] installed mmcv shim")


def run(env: ModelEnv, manifest: Path, limit: int = 0) -> Path:
    out = RUNS / f"{env.name}.json"
    python = Path(env.venv) / "bin" / "python"
    if not python.exists():
        RunResult(model=env.name, failed=True,
                  error="environment not built; run `setup` first").write(out)
        return out

    environ = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    if env.name == "metric3d_default_K":
        environ["DEPTHBENCH_REAL_K"] = "0"
    if env.name == "depth_pro":
        environ["DEPTHBENCH_DEPTHPRO_ROOT"] = str(Path(env.venv).parent)
    if env.name.startswith("yolo26"):
        environ["DEPTHBENCH_YOLO_WEIGHTS"] = env.variant

    command = [str(python), str(ROOT / "runners" / env.runner),
               "--manifest", str(manifest), "--out", str(out)]
    if limit:
        command += ["--limit", str(limit)]

    print(f"[{env.name}] running ...", flush=True)
    started = time.time()
    result = subprocess.run(command, env=environ, capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        tail = (result.stderr or result.stdout)[-1200:]
        RunResult(model=env.name, failed=True, error=tail).write(out)
        print(f"[{env.name}] FAILED after {time.time() - started:.0f}s")
    else:
        print(f"[{env.name}] done in {time.time() - started:.0f}s")
    return out


def score(manifest: Path, out: Path, depth_key: str) -> Path:
    from depthbench.metrics import score_run
    from depthbench.report import build_report

    loaded = Manifest.read(manifest)
    scores = []
    for name in MODELS:
        path = RUNS / f"{name}.json"
        if path.exists():
            scores.append(score_run(RunResult.read(path), loaded, depth_key=depth_key))
    if not scores:
        print("no runs found; run some models first")
        return out
    return build_report(scores, loaded, out, depth_key=depth_key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="depthbench")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest", help="build ground truth from nuScenes")
    m.add_argument("--dataroot", default="sessions/nuscenes")
    m.add_argument("--scenes", nargs="+",
                   default=["scene-0655", "scene-0757", "scene-0103", "scene-0553"])
    m.add_argument("--stride", type=int, default=2)
    m.add_argument("--max-range", type=float, default=50.0)
    m.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)

    for name in ("setup", "run"):
        p = sub.add_parser(name)
        p.add_argument("--model", action="append", choices=sorted(MODELS), default=None)
        p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        if name == "setup":
            p.add_argument("--force", action="store_true")
        else:
            p.add_argument("--limit", type=int, default=0)

    s = sub.add_parser("score")
    s.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    s.add_argument("--out", type=Path, default=Path("reports/depthbench.md"))
    s.add_argument("--depth", choices=("surface", "centroid"), default="surface")

    args = parser.parse_args(argv)

    if args.command == "manifest":
        from depthbench.dataset import build_manifest

        built = build_manifest(args.dataroot, args.scenes, max_range_m=args.max_range,
                               stride=args.stride)
        path = built.write(args.out)
        n = sum(len(s.objects) for s in built.samples)
        print(f"wrote {path}: {len(built.samples)} images, {n} objects")
        return 0

    if args.command in ("setup", "run"):
        names = args.model or list(MODELS)
        ok = True
        for name in names:
            env = MODELS[name]
            if args.command == "setup":
                ok &= setup(env, force=args.force)
            else:
                run(env, args.manifest, limit=args.limit)
        return 0 if ok else 1

    path = score(args.manifest, args.out, depth_key=f"{args.depth}_depth_m")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
