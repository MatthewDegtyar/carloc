"""Deeper repeat scan: lower threshold, verify many more candidates with ORB,
and report only NEW locations (excluding the known intro/Biscayne/roundabout)."""
import subprocess
import numpy as np
import cv2

SRC = "Miami Florida City Drive 4K -  Magic City Driving Tour.mp4"
GW, GH = 128, 72

def thumbs():
    p = subprocess.Popen(["ffmpeg","-v","error","-i",SRC,"-vf",
        f"fps=1,scale={GW}:{GH},format=gray","-f","rawvideo","-pix_fmt","gray","-"],
        stdout=subprocess.PIPE)
    fr=[]; n=GW*GH
    while True:
        b=p.stdout.read(n)
        if len(b)<n: break
        fr.append(np.frombuffer(b,np.uint8).reshape(GH,GW))
    p.wait(); return np.array(fr)

def desc(frames):
    D=[]
    for f in frames:
        up=f[:int(GH*0.62),:]; d=cv2.resize(up,(32,16)).astype(np.float32).ravel()
        d-=d.mean(); D.append(d/(np.linalg.norm(d)+1e-6))
    return np.array(D)

def frame_at(t):
    out=subprocess.run(["ffmpeg","-v","error","-ss",str(t),"-i",SRC,"-frames:v","1",
        "-f","image2pipe","-vcodec","png","-"],capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(out,np.uint8),cv2.IMREAD_GRAYSCALE)

KNOWN=[(0,25),(50*60,55*60),(22*60,23*60+30),(7*60,8*60+30),(17*60,18*60+30),
       (15*60,16*60+30),(81*60,83*60)]
def known(i,j):
    return any(a<=i<=b or a<=j<=b for a,b in KNOWN)

def main():
    F=thumbs(); N=len(F); D=desc(F)
    print(f"{N} frames")
    GAP=150; cand=[]
    for i in range(N):
        sims=D[i+GAP:]@D[i]
        if len(sims)==0: continue
        j=int(np.argmax(sims))+i+GAP; s=float(sims[j-(i+GAP)])
        if s>0.66 and not known(i,j): cand.append((s,i,j))
    cand.sort(reverse=True)
    kept=[]
    for s,i,j in cand:
        if any(abs(i-ki)<10 and abs(j-kj)<10 for _,ki,kj in kept): continue
        kept.append((s,i,j))
        if len(kept)>=120: break
    print(f"{len(kept)} new candidates to ORB-verify")
    orb=cv2.ORB_create(2500); bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    def inl(a,b):
        ka,da=orb.detectAndCompute(a,None); kb,db=orb.detectAndCompute(b,None)
        if da is None or db is None: return 0
        good=[m for m,n in bf.knnMatch(da,db,k=2) if m.distance<0.75*n.distance]
        if len(good)<12: return 0
        pa=np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pb=np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        _,mask=cv2.findHomography(pa,pb,cv2.RANSAC,5.0)
        return int(mask.sum()) if mask is not None else 0
    ver=[]
    for s,i,j in kept:
        n=inl(frame_at(i),frame_at(j))
        if n>=30:
            ver.append((n,i,j))
            print(f"  NEW REVISIT  {i//60:02d}:{i%60:02d} <-> {j//60:02d}:{j%60:02d}  inliers={n}",flush=True)
    print(f"\n{len(ver)} new verified repeats (excluding known spots)")
    import json
    json.dump([{"t1":i,"t2":j,"inliers":n} for n,i,j in sorted(ver,reverse=True)],
              open("reports/new_repeats.json","w"),indent=1)

if __name__=="__main__": main()
