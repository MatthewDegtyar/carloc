"""Before/after spread, boxing ONLY vehicles the tracker confirms as parked.

The quick version boxed every left-of-centre detection, which on a multi-lane
boulevard catches cars in the left travel lane. This runs the real pipeline --
associate detections into tracklets, keep those that triangulate as stationary
(the moving-vehicle slope test) -- and boxes only detections belonging to a kept
parked tracklet. Movers are not boxed, because they are not parked.
"""
import glob
import io
import math
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw
from rfdetr import RFDETRBase

from carloc.appearance import classify_colour, dominant_rgb
from carloc.rfdetr_detect import COCO_VEHICLES
from carloc.tracking import associate, triangulate

W, H = 1920, 1080
F_PX = 687.0
LATERAL_M = 7.0
model = RFDETRBase()


def dump(frames_dir, t0, t1, clip):
    """Detections per frame + camera along-position (scene motion, scaled)."""
    # scene motion for s_cam
    cap = cv2.VideoCapture(clip)
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    ok, prev = cap.read()
    pg = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    mags = []
    t = t0
    while t < t1:
        cap.set(cv2.CAP_PROP_POS_MSEC, (t + 0.25) * 1000)
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(pg, maxCorners=600, qualityLevel=0.01, minDistance=6, blockSize=7)
        v = 0.0
        if p0 is not None and len(p0) > 20:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, p0, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            if p1 is not None:
                s = st.ravel() == 1
                a, b = p0[s, 0], p1[s, 0]
                if len(a) > 20:
                    M, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if M is not None:
                        v = float(np.percentile(np.linalg.norm(b - ((a @ M[:, :2].T) + M[:, 2]), axis=1), 80))
        mags.append(v)
        pg = g
        t += 0.25
    cap.release()
    cum = np.concatenate([[0], np.cumsum(np.convolve(mags, np.ones(5) / 5, mode="same"))])
    cum = cum / cum[-1] * 333.0

    files = sorted(glob.glob(f"{frames_dir}/f_*.jpg"))
    dets = []
    for i, path in enumerate(files):
        arr = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        res = model.predict(arr, threshold=0.45)
        cls = np.array(res.class_id)
        box = np.array(res.xyxy)
        keep = np.isin(cls, list(COCO_VEHICLES))
        for (x1, y1, x2, y2), cid in zip(box[keep], cls[keep], strict=False):
            if (y2 - y1) < 80:
                continue
            cx = (x1 + x2) / 2
            bearing = math.degrees(math.atan((cx - W / 2) / F_PX))
            if bearing > -20:
                continue
            crop = arr[int(max(y1, 0)):int(y2), int(max(x1, 0)):int(x2)]
            dets.append({"frame": i, "s_cam": float(cum[min(i, len(cum) - 1)]),
                         "bearing": round(bearing, 2), "cx": float(cx), "cy": float((y1 + y2) / 2),
                         "bh": float(y2 - y1), "bw": float(x2 - x1),
                         "color": classify_colour(dominant_rgb(crop)),
                         "cls": COCO_VEHICLES.get(int(cid), "car"),
                         "bbox": [float(x1), float(y1), float(x2), float(y2)]})
    return dets


def parked_boxes(dets):
    """(frame -> list of (bbox, color)) for detections in a kept parked tracklet."""
    tracks = [t for t in associate(dets, W=W) if len(t.dets) >= 2]
    kept = {}
    for t in tracks:
        S, sig, k = triangulate(t, LATERAL_M)     # nan if moving (slope test) or no abeam
        if not math.isfinite(S):
            continue
        for d in t.dets:
            kept.setdefault(d["frame"], []).append((d["bbox"], d["color"]))
    return kept


def frame_gray(clip, t):
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", clip, "-frames:v", "1",
                          "-vf", "scale=1280:720", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True).stdout
    return cv2.cvtColor(np.array(Image.open(io.BytesIO(out)).convert("RGB")), cv2.COLOR_RGB2GRAY)


def main():
    P1, P2 = "video/pass1_1336.mp4", "video/pass2_3168.mp4"
    k1 = parked_boxes(dump("reports/p1f", 12, 42, P1))
    k2 = parked_boxes(dump("reports/p2f", 12, 33, P2))

    def draw(frames_dir, idx, kept):
        im = Image.open(f"{frames_dir}/f_{idx+1:04d}.jpg").convert("RGB")
        d = ImageDraw.Draw(im)
        for bbox, col in kept.get(idx, []):
            x1, y1, x2, y2 = bbox
            d.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(10, 14, 20), width=7)
            d.rectangle([x1, y1, x2, y2], outline=(99, 230, 196), width=5)
            d.text((x1 + 3, y1 - 16), col, fill=(99, 230, 196))
        return im, len(kept.get(idx, []))

    orb = cv2.ORB_create(2500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def inl(a, b):
        ka, da = orb.detectAndCompute(a, None)
        kb, db = orb.detectAndCompute(b, None)
        if da is None or db is None:
            return 0
        good = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            return 0
        pa = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pb = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 5.0)
        return int(mask.sum()) if mask is not None else 0

    rows = []
    for i1 in [8, 24, 40, 56, 72, 88, 104]:
        t1 = 12 + i1 / 4.0
        est = 12 + (t1 - 12) * (33 - 12) / (42 - 12)
        a = frame_gray(P1, t1)
        best = max((inl(a, frame_gray(P2, tt)), tt) for tt in np.arange(est - 2, est + 2.1, 0.5))
        i2 = int(round((best[1] - 12) * 4))
        pa, na = draw("reports/p1f", i1, k1)
        pb, nb = draw("reports/p2f", min(i2, 83), k2)
        rows.append((i1, na, nb, pa, pb))
        print(f"pass1 idx{i1} ({na} parked) <-> pass2 idx{i2} ({nb} parked)", flush=True)

    cw, ch = 640, 360
    sheet = Image.new("RGB", (cw * 2 + 18, len(rows) * (ch + 26) + 42), (13, 17, 23))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 12), "PARKED (tracker-confirmed stationary) - Biscayne 30.5 min apart - "
                      "pass 1 left · pass 2 right", fill=(230, 237, 243))
    for r, (i1, na, nb, pa, pb) in enumerate(rows):
        y = 40 + r * (ch + 26)
        sheet.paste(pa.resize((cw, ch)), (6, y))
        sheet.paste(pb.resize((cw, ch)), (cw + 12, y))
        sec = int(i1 / 4)
        dr.text((10, y + ch + 4), f"pass1 22:{26+sec//60:02d}:{sec%60:02d}  ·  {na} parked", fill=(120, 200, 255))
        dr.text((cw + 16, y + ch + 4), f"pass2 +30.5 min  ·  {nb} parked", fill=(255, 209, 102))
    sheet.save("reports/before_after.png")
    print("wrote reports/before_after.png")


if __name__ == "__main__":
    main()
