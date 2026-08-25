"""Minimal stand-in for the parts of mmcv 1.x that Metric3D imports.

Metric3D's `mono/utils/comm.py` does `from mmcv.utils import collect_env`, which is
the mmcv 1.x layout. mmcv 1.x needs compilation and has no macOS-ARM wheel; mmcv 2.x
moved the symbol to mmengine.

This shim exists because that import is NOT on the inference path. `collect_env`
and `get_git_hash` are environment logging, and `Config` appears only inside
`__main__` blocks in the repo. Nothing here participates in the model's forward
pass, so it changes what gets logged and not what gets predicted. Verified by
grepping every mmcv reference in the checkout before writing it.
"""

from mmcv import utils  # noqa: F401

__version__ = "1.x-shim"
