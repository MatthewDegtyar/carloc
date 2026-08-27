"""Stage 1: propose revisit candidates in the full drive via a global descriptor.

A driving tour rarely retraces itself, so the goal is just to surface the few
frame pairs, far apart in time, that look like the same place -- loop closures.
The descriptor leans on the stable part of a street scene (buildings and skyline,
upper frame) and ignores the road and traffic (lower frame), which change with
every pass. Cheap and permissive on purpose; ORB verification comes next.
"""
import subprocess
import sys

import numpy as np

SRC = "Miami Florida City Drive 4K -  Magic City Driving Tour.mp4"
GW, GH = 128, 72

def load_thumbs():
    cmd = ["ffmpeg", "-v", "error", "-i", SRC, "-vf", "fps=1,scale=%d:%d,format=gray" % (GW, GH),
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frames = []
    n = GW*GH
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(GH, GW))
    p.wait()
    return np.array(frames)

def descriptors(frames):
    import cv2
    D = []
    for f in frames:
        up = f[0:int(GH*0.62), :]                 # buildings + sky, drop road/traffic
        d = cv2.resize(up, (32, 16)).astype(np.float32).ravel()
        d -= d.mean()
        nrm = np.linalg.norm(d) + 1e-6
        D.append(d/nrm)
    return np.array(D)

def main():
    frames = load_thumbs()
    N = len(frames)
    print(f"{N} frames at 1 fps ({N/60:.1f} min)", flush=True)
    D = descriptors(frames)
    GAP = 120                                     # ignore pairs <2 min apart
    cand = []
    for i in range(N):
        sims = D[i+GAP:] @ D[i]
        if len(sims) == 0:
            continue
        j = int(np.argmax(sims)) + i + GAP
        s = float(sims.argmax() and sims[j-(i+GAP)] or (sims[0] if len(sims) else 0))
        s = float(sims[j-(i+GAP)])
        if s > 0.72:
            cand.append((s, i, j))
    cand.sort(reverse=True)
    # de-duplicate: keep candidates whose (i,j) aren't near an already-kept one
    kept = []
    for s, i, j in cand:
        if any(abs(i-ki) < 8 and abs(j-kj) < 8 for _, ki, kj in kept):
            continue
        kept.append((s, i, j))
        if len(kept) >= 60:
            break
    print(f"{len(kept)} candidate revisit pairs (sim>0.72, >2min apart):")
    for s, i, j in kept[:40]:
        print(f"  sim={s:.3f}   t1={i//60:02d}:{i%60:02d}  t2={j//60:02d}:{j%60:02d}  (dt={j-i}s)")
    import json
    json.dump([{"sim": s, "t1": i, "t2": j} for s, i, j in kept],
              open("reports/repeat_candidates.json", "w"), indent=1)

if __name__ == "__main__":
    main()
