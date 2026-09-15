from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from ..preprocessing import load_preprocessed

IM_MEAN=np.array([0.485,0.456,0.406],dtype=np.float32)
IM_STD=np.array([0.229,0.224,0.225],dtype=np.float32)

def _model(device):
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    model=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc=torch.nn.Identity(); model.eval().to(device)
    for p in model.parameters(): p.requires_grad_(False)
    return model

def _batch_tensor(paths):
    arr=[]
    for p in paths:
        rgb,_=load_preprocessed(p)
        x=(rgb-IM_MEAN)/IM_STD
        arr.append(np.transpose(x,(2,0,1)))
    return np.stack(arr).astype(np.float32)

def extract_spatial_chunked(manifest: pd.DataFrame, out_dir, batch_size=64, chunk_size=512, device=None):
    import torch
    out=Path(out_dir); chunks=out/"chunks"; chunks.mkdir(parents=True,exist_ok=True)
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    model=_model(device)
    n=len(manifest)
    for c0 in range(0,n,chunk_size):
        c1=min(n,c0+chunk_size); cp=chunks/f"chunk_{c0:06d}_{c1:06d}.npz"
        if cp.exists(): continue
        feats=[]; ids=[]
        for b0 in tqdm(range(c0,c1,batch_size),desc=f"ResNet {c0}:{c1}",leave=False):
            b1=min(c1,b0+batch_size); x=torch.from_numpy(_batch_tensor(manifest.filepath.iloc[b0:b1])).to(device)
            with torch.inference_mode(): z=model(x).detach().cpu().numpy()
            feats.append(z); ids.extend(manifest.image_id.iloc[b0:b1].tolist())
        tmp=cp.with_suffix(".tmp.npz"); np.savez_compressed(tmp,image_id=np.array(ids),X=np.vstack(feats).astype(np.float32)); tmp.replace(cp)
    final=out/"spatial_embeddings.npz"
    if not final.exists():
        X=[]; ids=[]
        for cp in sorted(chunks.glob("chunk_*.npz")):
            d=np.load(cp); X.append(d["X"]); ids.extend(d["image_id"].tolist())
        np.savez_compressed(final,image_id=np.array(ids),X=np.vstack(X).astype(np.float32))
    return final
