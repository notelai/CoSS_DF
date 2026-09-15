from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

def _nll(y,P): return -np.mean(np.log(P[np.arange(len(y)),y]))

def _draw_indices(df,rng,clustered):
    idx=[]
    for c,gc in df.groupby("class_id",sort=True):
        if clustered:
            units=gc.group_id.dropna().unique()
            chosen=rng.choice(units,size=len(units),replace=True)
            for u in chosen: idx.extend(gc.index[gc.group_id==u].tolist())
        else:
            ids=gc.index.to_numpy(); idx.extend(rng.choice(ids,size=len(ids),replace=True).tolist())
    return np.array(idx,dtype=int)

def paired_bootstrap(df,PA,PB,B=10000,seed=2026,clustered=False):
    # df must have RangeIndex aligned with P arrays.
    rng=np.random.default_rng(seed); y=df.class_id.to_numpy(); vals=[]
    for _ in range(B):
        ix=_draw_indices(df,rng,clustered)
        ya=y[ix]; a=PA[ix]; b=PB[ix]
        vals.append((f1_score(ya,a.argmax(1),average="macro")-f1_score(ya,b.argmax(1),average="macro"),_nll(ya,a)-_nll(ya,b)))
    z=np.asarray(vals)
    return {"delta_macro_f1_ci":[float(x) for x in np.quantile(z[:,0],[.025,.975])],"delta_nll_ci":[float(x) for x in np.quantile(z[:,1],[.025,.975])]}
