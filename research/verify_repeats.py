"""Stage 2: geometrically verify revisit candidates with ORB + RANSAC.

A global descriptor matches any two similar-looking downtown blocks. A real
revisit of the same place has many feature correspondences that agree on a single
geometric transform; a look-alike does not. So for each candidate pair, match ORB
features and count RANSAC-homography inliers -- that count is the truth signal.
"""
import json
import subprocess

import cv2
import numpy as np

SRC = "Miami Florida City Drive 4K -  Magic City Driving Tour.mp4"

def frame_at(t):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t), "-i", SRC, "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "png", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    arr = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_GRAYSCALE)
    return arr

def inliers(a, b, orb, bf):
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return 0
    matches = bf.knnMatch(da, db, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 15:
        return 0
    pa = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pb = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0

def main():
    cand = json.load(open("reports/repeat_candidates.json"))
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    verified = []
    for c in cand:
        t1, t2 = c["t1"], c["t2"]
        a, b = frame_at(t1), frame_at(t2)
        if a is None or b is None:
            continue
        n = inliers(a, b, orb, bf)
        tag = "REVISIT" if n >= 40 else ("maybe" if n >= 22 else "no")
        print(f"  t1={t1//60:02d}:{t1%60:02d} t2={t2//60:02d}:{t2%60:02d}  "
              f"sim={c['sim']:.3f}  inliers={n:3d}  {tag}", flush=True)
        if n >= 22:
            verified.append({"t1": t1, "t2": t2, "sim": c["sim"], "inliers": n})
    verified.sort(key=lambda v: -v["inliers"])
    json.dump(verified, open("reports/repeats_verified.json", "w"), indent=1)
    print(f"\n{len(verified)} geometrically-verified repeat pairs (>=22 inliers)")
    for v in verified[:12]:
        print(f"  {v['t1']//60:02d}:{v['t1']%60:02d} <-> {v['t2']//60:02d}:{v['t2']%60:02d}"
              f"   inliers={v['inliers']}")

if __name__ == "__main__":
    main()
