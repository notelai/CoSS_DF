from __future__ import annotations
from PIL import Image
import numpy as np


def load_preprocessed(path, size=224):
    with Image.open(path) as im:
        im=im.convert("RGB")
        w,h=im.size
        scale=size/min(w,h)
        nw,nh=round(w*scale),round(h*scale)
        im=im.resize((nw,nh),Image.Resampling.BICUBIC)
        left=max(0,(nw-size)//2); top=max(0,(nh-size)//2)
        im=im.crop((left,top,left+size,top+size))
        rgb=np.asarray(im,dtype=np.float32)/255.0
    y=0.299*rgb[...,0]+0.587*rgb[...,1]+0.114*rgb[...,2]
    return rgb,y.astype(np.float64)
