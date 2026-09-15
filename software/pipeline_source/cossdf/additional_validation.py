from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .datasets.splits import inner_splits
from .features.alternative_spectral import energies_to_ilr, clr_ilr_sanity
from .features.spectral import spectral_from_G
from .metrics import all_metrics
from .models.fusion import nll
from .models.transforms import RegularizedWhitener
from .analysis.bootstrap import paired_bootstrap
from .utils import atomic_json

CGRID=(.01,.1,1.,10.,100.)
RGRID=(4,6,8)
TOL=1e-12


def _align(model,P,K):
    out=np.zeros((len(P),K),float)
    for j,c in enumerate(model.classes_): out[:,int(c)]=P[:,j]
    return out


def _fit_lr(Xtr,ytr,Xte,C,K):
    mdl=make_pipeline(StandardScaler(),LogisticRegression(C=C,solver='lbfgs',max_iter=5000,random_state=2026))
    mdl.fit(Xtr,ytr); return _align(mdl[-1],mdl.predict_proba(Xte),K)


def _fit_spec_lr(Ztr,ytr,Zte,C,K,gamma=1e-6):
    wh=RegularizedWhitener(gamma).fit(Ztr)
    A=wh.transform(Ztr); B=wh.transform(Zte)
    mdl=LogisticRegression(C=C,solver='lbfgs',max_iter=5000,random_state=2026).fit(A,ytr)
    return _align(mdl,mdl.predict_proba(B),K)


def _oof_spec_lr(Z,y,dfdev,dataset,C,K,gamma=1e-6):
    P=np.zeros((len(y),K),float)
    for tr,va in inner_splits(dfdev,dataset): P[va]=_fit_spec_lr(Z[tr],y[tr],Z[va],C,K,gamma)
    return P


def _select_C_spec(Z,y,dfdev,dataset,K):
    rows=[]
    for C in CGRID:
        P=_oof_spec_lr(Z,y,dfdev,dataset,C,K); rows.append({'C':C,'nll':nll(P,y)})
    best=min(r['nll'] for r in rows); cand=[r for r in rows if abs(r['nll']-best)<=TOL]
    return sorted(cand,key=lambda r:r['C'])[0],pd.DataFrame(rows)


def _oof_lr(X,y,dfdev,dataset,C,K):
    P=np.zeros((len(y),K),float)
    for tr,va in inner_splits(dfdev,dataset): P[va]=_fit_lr(X[tr],y[tr],X[va],C,K)
    return P


def _select_C(X,y,dfdev,dataset,K):
    rows=[]
    for C in CGRID:
        P=_oof_lr(X,y,dfdev,dataset,C,K); rows.append({'C':C,'nll':nll(P,y)})
    best=min(r['nll'] for r in rows); cand=[r for r in rows if abs(r['nll']-best)<=TOL]
    return sorted(cand,key=lambda r:r['C'])[0],pd.DataFrame(rows)


def _select_dct_lr(G,y,dfdev,dataset,K,eta=1e-6):
    rows=[]; cache={}
    for R in RGRID:
        _,_,Z=spectral_from_G(G,R,eta)
        for C in CGRID:
            P=_oof_spec_lr(Z,y,dfdev,dataset,C,K); sc=nll(P,y); rows.append({'R':R,'C':C,'nll':sc}); cache[(R,C)]=Z
    best=min(r['nll'] for r in rows); cand=[r for r in rows if abs(r['nll']-best)<=TOL]
    chosen=sorted(cand,key=lambda r:(r['R'],r['C']))[0]
    return chosen,pd.DataFrame(rows),cache[(chosen['R'],chosen['C'])]


def _prob_cols(method,K): return [f'p__{method}__{k}' for k in range(K)]


def run_additional_validation(dataset,manifest,folds,G,E_dwt,E_gabor,Xeff,Xvit,old_directional_fusion_controls,out_dir:Path,bootstrap_B=10000):
    if dataset not in {'kth_tips2b','fmd','kylberg'}: raise RuntimeError('additional_validation primary datasets only')
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    K=int(manifest.class_id.nunique()); rows=[]; sels=[]; timing=[]

    Zdwt_all=energies_to_ilr(E_dwt)[1]; Zgabor_all=energies_to_ilr(E_gabor)[1]

    for fold in sorted(folds.outer_fold.unique()):
        fp=out_dir/f'fold_{int(fold)}.npz'; sp=out_dir/f'fold_{int(fold)}_selected.json'
        if fp.exists() and sp.exists():
            d=np.load(fp); te=d['test_index']; payload={k:d[k] for k in d.files if k not in {'test_index','y_true'}}; info=json.loads(sp.read_text())
        else:
            te=np.flatnonzero(folds.outer_fold.values==fold); tr=np.flatnonzero(folds.outer_fold.values!=fold)
            ytr=manifest.class_id.values[tr]; yte=manifest.class_id.values[te]; dfdev=manifest.iloc[tr].reset_index(drop=True)
            t0=time.perf_counter()
            # fixed DWT and Gabor energy compositions -> ILR -> logistic; tune C only.
            ds,dt=_select_C_spec(Zdwt_all[tr],ytr,dfdev,dataset,K); gs,gt=_select_C_spec(Zgabor_all[tr],ytr,dfdev,dataset,K)
            dt.to_csv(out_dir/f'fold_{int(fold)}_dwt_C_candidates.csv',index=False); gt.to_csv(out_dir/f'fold_{int(fold)}_gabor_C_candidates.csv',index=False)
            P_DWT=_fit_spec_lr(Zdwt_all[tr],ytr,Zdwt_all[te],float(ds['C']),K)
            P_GABOR=_fit_spec_lr(Zgabor_all[tr],ytr,Zgabor_all[te],float(gs['C']),K)
            # modern frozen embeddings + logistic; same inner NLL C selection.
            es,et=_select_C(Xeff[tr],ytr,dfdev,dataset,K); vs,vt=_select_C(Xvit[tr],ytr,dfdev,dataset,K)
            et.to_csv(out_dir/f'fold_{int(fold)}_effnet_C_candidates.csv',index=False); vt.to_csv(out_dir/f'fold_{int(fold)}_vit_C_candidates.csv',index=False)
            P_EFF=_fit_lr(Xeff[tr],ytr,Xeff[te],float(es['C']),K)
            P_VIT=_fit_lr(Xvit[tr],ytr,Xvit[te],float(vs['C']),K)
            # DCT compositional logistic baseline, selected within each outer dev fold.
            dctsel,dcttab,Zdct_tr=_select_dct_lr(G[tr],ytr,dfdev,dataset,K)
            dcttab.to_csv(out_dir/f'fold_{int(fold)}_dct_lr_candidates.csv',index=False)
            _,_,Zdct_te=spectral_from_G(G[te],int(dctsel['R']),1e-6)
            P_DCT=_fit_spec_lr(Zdct_tr,ytr,Zdct_te,float(dctsel['C']),K)
            payload={'P_DCT_ILR_LR':P_DCT,'P_DWT_ILR_LR':P_DWT,'P_GABOR_ILR_LR':P_GABOR,'P_EFFICIENTNET_B0_LR':P_EFF,'P_VIT_B16_LR':P_VIT}
            np.savez_compressed(fp,test_index=te,y_true=yte,**payload)
            info={'outer_fold':int(fold),'dct_R':int(dctsel['R']),'dct_C':float(dctsel['C']),'dwt_C':float(ds['C']),'gabor_C':float(gs['C']),'effnet_C':float(es['C']),'vit_C':float(vs['C']),'modeling_seconds':float(time.perf_counter()-t0)}
            atomic_json(sp,info)
        sels.append(info); yte=manifest.class_id.values[te]
        for pos,idx in enumerate(te):
            rec={'image_id':manifest.image_id.iloc[idx],'class_id':int(yte[pos]),'outer_fold':int(fold)}
            for key,P in payload.items():
                name=key[2:] if key.startswith('P_') else key
                for k in range(K): rec[f'p__{name}__{k}']=float(P[pos,k])
            rows.append(rec)

    pred=pd.DataFrame(rows).sort_values('image_id').reset_index(drop=True); pred.to_csv(out_dir/'additional_validation_outer_predictions.csv',index=False)
    pd.DataFrame(sels).to_csv(out_dir/'additional_validation_selected.csv',index=False)
    y=pred.class_id.to_numpy(); methods=['DCT_ILR_LR','DWT_ILR_LR','GABOR_ILR_LR','EFFICIENTNET_B0_LR','VIT_B16_LR']
    mets=[]
    for m in methods: mets.append({'dataset':dataset,'method':m,**all_metrics(y,P=pred[_prob_cols(m,K)].to_numpy())})
    met=pd.DataFrame(mets); met.to_csv(out_dir/'additional_validation_metrics.csv',index=False)

    # Secondary paired comparisons. Nominal intervals only.
    meta=manifest.set_index('image_id').loc[pred.image_id].reset_index(); clustered=dataset in {'kth_tips2b','kylberg'}
    effects=[]
    pairs=[('DWT_ILR_LR','DCT_ILR_LR'),('GABOR_ILR_LR','DCT_ILR_LR')]
    # Compare modern baselines with the existing strong ResNet18 logistic C3S control.
    ov=old_directional_fusion_controls.set_index('image_id').loc[pred.image_id].reset_index()
    P_C3S=ov[[f'p__C3S__{k}' for k in range(K)]].to_numpy()
    for a,b in pairs:
        PA=pred[_prob_cols(a,K)].to_numpy(); PB=pred[_prob_cols(b,K)].to_numpy(); boot=paired_bootstrap(meta,PA,PB,B=bootstrap_B,seed=2026,clustered=clustered)
        ma=all_metrics(y,P=PA); mb=all_metrics(y,P=PB)
        effects.append({'dataset':dataset,'comparison':f'{a}_minus_{b}','delta_macro_f1':ma['macro_f1']-mb['macro_f1'],'delta_nll':ma['nll']-mb['nll'],**boot})
    for a in ['EFFICIENTNET_B0_LR','VIT_B16_LR']:
        PA=pred[_prob_cols(a,K)].to_numpy(); boot=paired_bootstrap(meta,PA,P_C3S,B=bootstrap_B,seed=2026,clustered=clustered)
        ma=all_metrics(y,P=PA); mb=all_metrics(y,P=P_C3S)
        effects.append({'dataset':dataset,'comparison':f'{a}_minus_RESNET18_C3S','delta_macro_f1':ma['macro_f1']-mb['macro_f1'],'delta_nll':ma['nll']-mb['nll'],**boot})
    eff=pd.DataFrame(effects); eff.to_csv(out_dir/'additional_validation_effects.csv',index=False)
    return met,eff,pd.DataFrame(sels)
