"""Apple Depth Pro. Estimates its own focal length, so no intrinsics are supplied.

Prefers the official `depth_pro` package; falls back to the HF port if the git
install did not provide weights. Which path ran is recorded in the result notes,
because they are not guaranteed to be the same model revision.
"""

import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402


def _load_official(device):
    # The package resolves its checkpoint relative to the working directory
    # ("./checkpoints/depth_pro.pt"), so without this it raises FileNotFoundError
    # and the runner silently falls back to the HF port -- a different artefact.
    checkpoints = os.environ.get("DEPTHBENCH_DEPTHPRO_ROOT")
    if checkpoints and os.path.isdir(checkpoints):
        os.chdir(checkpoints)

    import depth_pro

    model, transform = depth_pro.create_model_and_transforms(device=device)
    model.eval()

    def predict(path):
        image, _, f_px = depth_pro.load_rgb(path)
        out = model.infer(transform(image), f_px=f_px)
        return np.asarray(out["depth"].detach().cpu().numpy(), dtype=np.float64)

    return predict, "official depth_pro package"


def _load_hf(device):
    import torch
    from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

    processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device).eval()

    def predict(path):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        post = processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )
        return np.asarray(post[0]["predicted_depth"].cpu().numpy(), dtype=np.float64)

    return predict, "huggingface apple/DepthPro-hf"


def main():
    args = _common.parse_args()
    manifest = _common.load_manifest(args.manifest)
    samples = manifest["samples"][: args.limit] if args.limit else manifest["samples"]
    device = args.device or _common.pick_device()

    try:
        predict, source = _load_official(device)
    except Exception as exc:  # noqa: BLE001 - fall back and say so
        try:
            predict, source = _load_hf(device)
            source += f" (official package unavailable: {type(exc).__name__})"
        except Exception as exc2:  # noqa: BLE001
            _common.write_result(args.out, "Depth Pro", "", [], float("nan"), device,
                                 failed=True, error=f"{exc} / {exc2}")
            return

    predictions, started = [], time.time()
    for sample in samples:
        depth = predict(sample["image"])
        for obj in sample["objects"]:
            predictions.append({
                "obj_id": obj["obj_id"], "image": sample["image"],
                "pred_depth_m": _common.sample_depth(
                    depth, obj["bbox"], (sample["width"], sample["height"])
                ),
            })
    _common.write_result(
        args.out, "Depth Pro", "", predictions,
        (time.time() - started) / max(len(samples), 1), device,
        notes=f"{source}; focal length estimated by the model, intrinsics not supplied.",
    )


if __name__ == "__main__":
    main()
