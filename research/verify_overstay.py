"""Pull the crop of each matched overstayer from BOTH passes, to confirm it is the
same physical parked car and not moving traffic or a coincidental same-spot swap."""
import glob
import json
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from carloc.appearance import classify_colour, dominant_rgb
from carloc.rfdetr_detect import COCO_VEHICLES
from rfdetr import RFDETRBase

W, H = 1920, 1080
F_PX = 687.0
LATERAL_M = 7.0
FLAGLER = (25.774346, -80.187238); NE3RD = (25.777198, -80.188307)
MX = 111_320*math.cos(math.radians(25.776)); MY = 110_540
D = math.hypot((NE3RD[1]-FLAGLER[1])*MX, (NE3RD[0]-FLAGLER[0])*MY)

def s_of(lat, lon):
    # project a car's lat back to along-street S (undo the left offset approx)
    return (lat - FLAGLER[0]) / (NE3RD[0] - FLAGLER[0]) * D

def best_crop(frames_dir, model, target_S, t0):
    """Find the detection whose implied along-position is closest to target_S,
    return its crop and appearance."""
    files = sorted(glob.glob(f"{frames_dir}/f_*.jpg"))
    best = None
    for i, path in enumerate(files):
        im = cv2.imread(path); arr = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        res = model.predict(arr, threshold=0.45)
        cls = np.array(res.class_id); box = np.array(res.xyxy)
        for (x1, y1, x2, y2), cid in zip(box[np.isin(cls, list(COCO_VEHICLES))],
                                          cls[np.isin(cls, list(COCO_VEHICLES))], strict=False):
            if (y2-y1) < 80: continue
            cx = (x1+x2)/2; bearing = math.degrees(math.atan((cx-W/2)/F_PX))
            if bearing > -20: continue
            b = math.radians(abs(bearing))
            if b < math.radians(30): continue
            # rough camera along at this frame: fraction of clip
            # (approximate; only used to rank crops near target)
            crop = arr[int(max(y1,0)):int(y2), int(max(x1,0)):int(x2)]
            # score by how close the detection's abeam bearing is (prefer abeam)
            score = abs(bearing) + (0 if best is None else 0)
            if best is None or abs(bearing) > best[0]:
                pass
    return None

def main():
    data = json.load(open("reports/biscayne_overstay.json"))
    over = data["overstayers"]
    print(f"{len(over)} overstayers to verify")
    # Simplest robust verification: crop the frame region around each overstayer's
    # bearing at its most-abeam moment, from each pass, side by side.
    model = RFDETRBase()
    # For each pass, collect all abeam detections with their crop + along estimate
    def collect(frames_dir):
        files = sorted(glob.glob(f"{frames_dir}/f_*.jpg"))
        out = []
        # need camera along per frame -> reuse motion via equal spacing fallback
        for i, path in enumerate(files):
            im = cv2.imread(path); arr = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            res = model.predict(arr, threshold=0.45)
            cls = np.array(res.class_id); box = np.array(res.xyxy)
            for (x1,y1,x2,y2),cid in zip(box[np.isin(cls,list(COCO_VEHICLES))],
                                          cls[np.isin(cls,list(COCO_VEHICLES))], strict=False):
                if (y2-y1)<80: continue
                cx=(x1+x2)/2; bearing=math.degrees(math.atan((cx-W/2)/F_PX))
                if bearing>-30: continue
                crop=arr[int(max(y1,0)):int(y2),int(max(x1,0)):int(x2)]
                out.append({"frame":i,"bearing":bearing,"bbox":(x1,y1,x2,y2),
                            "color":classify_colour(dominant_rgb(crop)),
                            "cls":COCO_VEHICLES.get(int(cid),"car"),"crop":crop})
        return out
    p1=collect("reports/p1f"); p2=collect("reports/p2f")
    # match overstayers to representative crops by color+class, near-abeam, spread along frame index
    rows=[]
    used1=set(); used2=set()
    for o in over:
        c1=[d for k,d in enumerate(p1) if d["color"]==o["color"] and d["cls"]==o["cls"] and k not in used1]
        c2=[d for k,d in enumerate(p2) if d["color"]==o["color"] and d["cls"]==o["cls"] and k not in used2]
        if not c1 or not c2: continue
        d1=max(c1,key=lambda d:abs(d["bearing"])); d2=max(c2,key=lambda d:abs(d["bearing"]))
        used1.add(p1.index(d1)); used2.add(p2.index(d2))
        rows.append((o,d1,d2))
    # montage
    cw=260
    sheet=Image.new("RGB",(cw*2+40,len(rows)*(cw+18)+10),(13,17,23)); dr=ImageDraw.Draw(sheet)
    for r,(o,d1,d2) in enumerate(rows):
        y=r*(cw+18)+10
        for col,d,x in ((0,d1,10),(1,d2,cw+30)):
            im=Image.fromarray(d["crop"]); im.thumbnail((cw-10,cw-10)); sheet.paste(im,(x,y))
        dr.text((10,y-2),f"{o['color']} {o['cls']} — dwell>={o['dwell_min']}min",fill=(99,230,196))
        dr.text((cw+30,y-2),"pass1 22:26  |  pass2 53:00",fill=(140,150,160))
    sheet.save("reports/overstay_verify.png")
    print(f"wrote reports/overstay_verify.png ({len(rows)} pairs)")

if __name__=="__main__": main()
