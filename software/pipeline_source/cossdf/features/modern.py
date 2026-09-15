from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ..preprocessing import load_preprocessed

IM_MEAN=np.array([0.485,0.456,0.406],dtype=np.float32)
IM_STD=np.array([0.229,0.224,0.225],dtype=np.float32)


def _batch_tensor(paths):
    arr=[]
    for p in paths:
        rgb,_=load_preprocessed(p)
        x=(rgb-IM_MEAN)/IM_STD
        arr.append(np.transpose(x,(2,0,1)))
    return np.stack(arr).astype(np.float32)


def _model(backbone,device):
    import torch
    if backbone=='efficientnet_b0':
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model=efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier=torch.nn.Identity(); dim=1280
    elif backbone=='vit_b_16':
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads=torch.nn.Identity(); dim=768
    else:
        raise ValueError(backbone)
    model.eval().to(device)
    for p in model.parameters(): p.requires_grad_(False)
    return model,dim


def extract_backbone_chunked(manifest: pd.DataFrame,out_dir,backbone,batch_size=32,chunk_size=256,device=None):
    import torch
    out=Path(out_dir); chunks=out/'chunks'; chunks.mkdir(parents=True,exist_ok=True)
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model,dim=_model(backbone,device); n=len(manifest)
    for c0 in range(0,n,chunk_size):
        c1=min(n,c0+chunk_size); cp=chunks/f'chunk_{c0:06d}_{c1:06d}.npz'
        if cp.exists(): continue
        feats=[]; ids=[]
        for b0 in tqdm(range(c0,c1,batch_size),desc=f'{backbone} {c0}:{c1}',leave=False):
            b1=min(c1,b0+batch_size)
            x=torch.from_numpy(_batch_tensor(manifest.filepath.iloc[b0:b1])).to(device)
            with torch.inference_mode(): z=model(x).detach().cpu().numpy()
            if z.ndim!=2 or z.shape[1]!=dim: raise RuntimeError(f'Unexpected {backbone} embedding shape {z.shape}')
            feats.append(z); ids.extend(manifest.image_id.iloc[b0:b1].tolist())
        tmp=cp.with_suffix('.tmp.npz'); np.savez_compressed(tmp,image_id=np.array(ids),X=np.vstack(feats).astype(np.float32)); tmp.replace(cp)
    final=out/'embeddings.npz'
    if not final.exists():
        ids=[]; X=[]
        for cp in sorted(chunks.glob('chunk_*.npz')):
            d=np.load(cp); ids.extend(d['image_id'].tolist()); X.append(d['X'])
        np.savez_compressed(final,image_id=np.array(ids),X=np.vstack(X).astype(np.float32))
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return final
