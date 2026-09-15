from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd
from .bootstrap import paired_bootstrap
from ..features.spectral import block_dct_coordinate_energy, spectral_from_G, energy_to_bands, smooth_composition, ilr
from ..preprocessing import load_preprocessed
from ..models.transforms import SpatialPCAWhitener, RegularizedWhitener
from ..models.density import IsotropicClassKDE
from ..models.fusion import logop
from ..metrics import all_metrics


def affine_verification(manifest,out_csv,eta=1e-6,m=8):
    transforms=[(a,b) for a in [.5,.75,1.25,1.5] for b in [-.2,-.1,.1,.2]]
    rows=[]
    for _,r in manifest.iterrows():
        _,y=load_preprocessed(r.filepath); G=block_dct_coordinate_energy(y,m); E0=energy_to_bands(G,8); p0=smooth_composition(E0,eta); z0=ilr(p0); norm=max(np.linalg.norm(E0),1e-300)
        for a,b in transforms:
            yp=a*y+b; Gp=block_dct_coordinate_energy(yp,m); Ep=energy_to_bands(Gp,8); pp=smooth_composition(Ep,eta); zp=ilr(pp)
            rows.append({"image_id":r.image_id,"a":a,"b":b,"delta_p_inf":float(np.max(np.abs(pp-p0))),"delta_ilr_l2":float(np.linalg.norm(zp-z0)),"delta_E_rel":float(np.linalg.norm(Ep-E0)/norm)})
    df=pd.DataFrame(rows); Path(out_csv).parent.mkdir(parents=True,exist_ok=True); df.to_csv(out_csv,index=False); return df

def condition_metrics(pred_df,method="CoSS_DF"):
    K=len([c for c in pred_df if c.startswith(f"p__{method}__")])
    P=pred_df[[f"p__{method}__{k}" for k in range(K)]].values; y=pred_df.class_id.values
    rows=[]
    for col in ["scale","pose","illumination"]:
        for value,g in pred_df.dropna(subset=[col]).groupby(col):
            ix=g.index.to_numpy(); rows.append({"condition":col,"value":value,"n":len(ix),**all_metrics(y[ix],P=P[ix])})
    return pd.DataFrame(rows)

def _load_selected(selected_csv):
    return pd.read_csv(selected_csv)

def variant_spectral_outer(dataset,manifest,folds,Xsp,G_variant,selected_configs,eta=1e-6,gamma=1e-6):
    """Evaluate a spectral numerical/grid variant with primary selected hyperparameters; no retuning."""
    selected=pd.read_csv(selected_configs); K=manifest.class_id.nunique(); allrows=[]
    for fold in sorted(folds.outer_fold.unique()):
        te=np.flatnonzero(folds.outer_fold.values==fold); tr=np.flatnonzero(folds.outer_fold.values!=fold); ytr=manifest.class_id.values[tr]
        row=selected[selected.outer_fold==fold].iloc[0]
        ds=int(row["joint.d_s"]); hs=float(row["joint.h_s"]); R=int(row["joint.R"]); hf=float(row["joint.h_f"]); lam=float(row["joint.lambda"])
        st=SpatialPCAWhitener(ds).fit(Xsp[tr]); As=st.transform(Xsp[tr]); Bs=st.transform(Xsp[te]); Ps=IsotropicClassKDE(hs).fit(As,ytr).predict_proba(Bs)
        Ztr=spectral_from_G(G_variant[tr],R,eta)[2]; Zte=spectral_from_G(G_variant[te],R,eta)[2]
        ft=RegularizedWhitener(gamma).fit(Ztr); Af=ft.transform(Ztr); Bf=ft.transform(Zte); Pf=IsotropicClassKDE(hf).fit(Af,ytr).predict_proba(Bf)
        P=logop(Ps,Pf,lam)
        for j,idx in enumerate(te): allrows.append((idx,P[j]))
    allrows=sorted(allrows,key=lambda x:x[0]); return np.vstack([p for _,p in allrows])

def affine_verification_chunked(manifest,out_dir,eta=1e-6,m=8,chunk_size=64):
    out=Path(out_dir); chunks=out/"affine_chunks"; chunks.mkdir(parents=True,exist_ok=True)
    transforms=[(a,b) for a in [.5,.75,1.25,1.5] for b in [-.2,-.1,.1,.2]]
    for c0 in range(0,len(manifest),chunk_size):
        c1=min(len(manifest),c0+chunk_size); cp=chunks/f"chunk_{c0:06d}_{c1:06d}.csv"
        if cp.exists(): continue
        rows=[]
        for i in range(c0,c1):
            r=manifest.iloc[i]; _,y=load_preprocessed(r.filepath); G=block_dct_coordinate_energy(y,m); E0=energy_to_bands(G,8); p0=smooth_composition(E0,eta); z0=ilr(p0); norm=max(np.linalg.norm(E0),1e-300)
            for a,b in transforms:
                Gp=block_dct_coordinate_energy(a*y+b,m); Ep=energy_to_bands(Gp,8); pp=smooth_composition(Ep,eta); zp=ilr(pp)
                rows.append({"image_id":r.image_id,"a":a,"b":b,"delta_p_inf":float(np.max(np.abs(pp-p0))),"delta_ilr_l2":float(np.linalg.norm(zp-z0)),"delta_E_rel":float(np.linalg.norm(Ep-E0)/norm)})
        pd.DataFrame(rows).to_csv(cp,index=False)
    final=out/"affine_verification.csv"
    pd.concat([pd.read_csv(p) for p in sorted(chunks.glob("chunk_*.csv"))],ignore_index=True).to_csv(final,index=False)
    return pd.read_csv(final)
