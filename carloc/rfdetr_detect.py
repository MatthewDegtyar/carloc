"""Back-compat shim. Detection now lives in :mod:`carloc.detect`; this re-exports
the names older code imported from here."""

from carloc.detect import (  # noqa: F401
    COCO_VEHICLES,
    NMS_IOU,
    OVERLAP,
    THRESHOLD,
    TILE,
    Detection,
    RFDETRDetector,
    _suppress,
)
