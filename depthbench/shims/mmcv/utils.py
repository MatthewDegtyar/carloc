"""mmcv.utils symbols Metric3D imports, delegated to mmengine where it has them."""


def collect_env():
    try:
        from mmengine.utils.dl_utils import collect_env as _collect

        return _collect()
    except Exception:  # noqa: BLE001 - logging only
        return {"note": "mmcv shim; environment collection unavailable"}


def get_git_hash(*args, **kwargs):
    try:
        from mmengine.utils import get_git_hash as _hash

        return _hash(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return "unknown"


try:  # Config is used only in the repo's __main__ blocks, never in inference.
    from mmengine import Config, DictAction  # noqa: F401
except Exception:  # noqa: BLE001
    class Config:  # type: ignore[no-redef]
        pass

    class DictAction:  # type: ignore[no-redef]
        pass
