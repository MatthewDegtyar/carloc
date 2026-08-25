"""Export YOLO11n to CoreML for `CoreMLDetector`.

Run this in a SEPARATE environment. It needs ultralytics and torch, which the
project does not depend on -- the export is a one-time offline step and only the
resulting .mlpackage is needed at runtime.

    uv venv --python 3.12 /tmp/yolo-export/.venv
    VIRTUAL_ENV=/tmp/yolo-export/.venv uv pip install ultralytics coremltools "numpy<2"
    /tmp/yolo-export/.venv/bin/python scripts/export_yolo.py

The `numpy<2` pin is not optional. coremltools' Torch converter calls `int()` on
shape-(1,) arrays, which numpy 2 rejects with

    TypeError: only 0-dimensional arrays can be converted to Python scalars

That error surfaces deep in the graph converter and looks like a torch version
problem; it is not. Pinning numpy fixes it on any tested torch.
"""

import shutil
import sys
from pathlib import Path

OUT_DIR = Path("models")


def main() -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print(__doc__)
        return 1

    model = YOLO("yolo11n.pt")
    exported = Path(model.export(format="coreml", nms=True, imgsz=640, half=False))
    OUT_DIR.mkdir(exist_ok=True)
    target = OUT_DIR / exported.name
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(exported), target)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
