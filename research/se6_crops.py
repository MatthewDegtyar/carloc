"""Re-run the SE 6th detection, but keep each kept car's source crop, to see
what RF-DETR actually fired on. Same placement + clustering as se6_detect.py."""
import glob
import json
import math

import numpy as np
from PIL import Image

from carloc.appearance import SWATCH, classify_colour, dominant_rgb
from carloc.rfdetr_detect import COCO_VEHICLES

W, H = 1920, 1080
F_PX = 687.0
LATERAL_M = 7.0
MIN_BEARING_DEG = 22.0
MIN_BOX_H = 90
CLUSTER_M = 3.5


def main():
    from rfdetr import RFDETRBase
    with open("reports/se6_track.json") as fh: track=json.load(fh)
    ts=np.array([r["t"] for r in track])
    model=RFDETRBase()
    files=sorted(glob.glob("reports/se6frames/f_*.jpg"))
    mx=111_320*math.cos(math.radians(25.768)); my=110_540
    raw=[]
    for i,path in enumerate(files):
        t=420.0+i/4.0
        k=int(np.clip(np.searchsorted(ts,t),0,len(track)-1)); fix=track[k]
        image=Image.open(path).convert("RGB"); arr=np.asarray(image)
        det=model.predict(image,threshold=0.5)
        cls=np.array(det.class_id); box=np.array(det.xyxy); conf=np.array(det.confidence)
        keep=np.isin(cls,list(COCO_VEHICLES))
        for (x1,y1,x2,y2),cid,sco in zip(box[keep],cls[keep],conf[keep]):
            if (y2-y1)<MIN_BOX_H: continue
            cx=(x1+x2)/2; bearing=math.degrees(math.atan((cx-W/2)/F_PX))
            if bearing>-MIN_BEARING_DEG: continue
            along=LATERAL_M/math.tan(math.radians(-bearing))
            if along>45: continue
            crop=arr[int(max(y1,0)):int(y2),int(max(x1,0)):int(x2)]
            rgb=dominant_rgb(crop) if crop.size else (110,112,115)
            hd=math.radians(fix["heading_deg"])
            dn=along*math.cos(hd)+LATERAL_M*math.cos(hd-math.pi/2)
            de=along*math.sin(hd)+LATERAL_M*math.sin(hd-math.pi/2)
            raw.append({"video_t":round(t,2),"lat":fix["lat"]+dn/my,"lon":fix["lon"]+de/mx,
                "sigma_along_m":fix["sigma_along_m"],"color":classify_colour(rgb),
                "conf":float(sco),"cls":COCO_VEHICLES.get(int(cid),"car"),
                "bbox":[float(x1),float(y1),float(x2),float(y2)],
                "aspect":round((x2-x1)/(y2-y1),2),"path":path,"rgb":list(rgb)})
    raw.sort(key=lambda r:r["sigma_along_m"])
    cars=[]
    for r in raw:
        q=np.array([r["lon"]*mx,r["lat"]*my])
        if any(np.hypot(c["lon"]*mx-q[0],c["lat"]*my-q[1])<CLUSTER_M for c in cars): continue
        cars.append(r)
    # save crops + contact sheet grouped by colour
    order=["black","grey","silver","white","red","orange","tan","green","blue","purple"]
    cars.sort(key=lambda c:(order.index(c["color"]),-c["conf"]))
    cell=200; cols=8; rows=(len(cars)+cols-1)//cols
    sheet=Image.new("RGB",(cols*cell,rows*(cell+30)),(13,17,23))
    from PIL import ImageDraw
    d=ImageDraw.Draw(sheet)
    for idx,c in enumerate(cars):
        im=Image.open(c["path"]).convert("RGB")
        x1,y1,x2,y2=[int(v) for v in c["bbox"]]
        pad=8
        crop=im.crop((max(0,x1-pad),max(0,y1-pad),min(W,x2+pad),min(H,y2+pad)))
        crop.thumbnail((cell-8,cell-8))
        cxp=(idx%cols)*cell; cyp=(idx//cols)*(cell+30)
        sheet.paste(crop,(cxp+4,cyp+4))
        d.rectangle([cxp+2,cyp+2,cxp+cell-2,cyp+cell-2],outline=SWATCH.get(c["color"],"#888"),width=2)
        d.text((cxp+5,cyp+cell+2),f"{c['color']} {c['cls']} c={c['conf']:.2f} a={c['aspect']}",fill=(200,210,220))
    sheet.save("reports/se6_crops.png")
    json.dump(cars,open("reports/se6_crops.json","w"),indent=1,default=float)
    from collections import Counter
    print(f"{len(cars)} cars")
    print("by colour:",dict(Counter(c['color'] for c in cars)))
    print("conf by colour:")
    for col in order:
        cc=[c['conf'] for c in cars if c['color']==col]
        if cc: print(f"  {col:7s} n={len(cc):2d} conf {min(cc):.2f}-{max(cc):.2f} median {sorted(cc)[len(cc)//2]:.2f}")
    print("wrote reports/se6_crops.png")

if __name__=="__main__": main()
