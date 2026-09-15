from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from ..models.baselines import lbp_hist_from_path, LBP_SPECS

def extract_lbp_chunked(manifest: pd.DataFrame,out_dir,chunk_size=256):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); finals=[]
    for P,R in LBP_SPECS:
        sd=out/f"P{P}_R{R}"; chunks=sd/"chunks"; chunks.mkdir(parents=True,exist_ok=True); n=len(manifest)
        for c0 in range(0,n,chunk_size):
            c1=min(n,c0+chunk_size); cp=chunks/f"chunk_{c0:06d}_{c1:06d}.npz"
            if cp.exists(): continue
            X=[]; ids=[]
            for i in tqdm(range(c0,c1),desc=f"LBP P{P} R{R} {c0}:{c1}",leave=False):
                X.append(lbp_hist_from_path(manifest.filepath.iloc[i],P,R)); ids.append(manifest.image_id.iloc[i])
            tmp=cp.with_suffix(".tmp.npz"); np.savez_compressed(tmp,image_id=np.array(ids),X=np.vstack(X)); tmp.replace(cp)
        final=sd/"features.npz"
        if not final.exists():
            X=[]; ids=[]
            for cp in sorted(chunks.glob("chunk_*.npz")):
                d=np.load(cp); X.append(d["X"]); ids.extend(d["image_id"].tolist())
            np.savez_compressed(final,image_id=np.array(ids),X=np.vstack(X))
        finals.append(final)
    return finals
