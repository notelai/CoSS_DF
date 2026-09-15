from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .datasets.splits import inner_splits
from .features.spectral import spectral_from_G, helmert_basis
from .models.transforms import RegularizedWhitener
from .models.density import IsotropicClassKDE
from .models.fusion import nll
from .metrics import all_metrics
from .analysis.bootstrap import paired_bootstrap
from .utils import atomic_json

CGRID=(.01,.1,1,10,100)
TOL=1e-12


def _prob_cols(pred: pd.DataFrame, method: str, K: int):
    return [f"p__{method}__{k}" for k in range(K)]


def _linear_simplex_coords(G, R, eta=1e-6):
    """Euclidean control: same closed composition, Helmert projection, no log-ratio."""
    _,P,_=spectral_from_G(G,R,eta)
    V=helmert_basis(R)
    # P@V == (P-u_R)@V because u_R@V = 0.
    return P@V


def _fit_b4w_matched(Gtr,ytr,Gte,R,h,eta=1e-6,gamma=1e-6):
    Ztr=_linear_simplex_coords(Gtr,R,eta)
    Zte=_linear_simplex_coords(Gte,R,eta)
    tf=RegularizedWhitener(gamma).fit(Ztr)
    A=tf.transform(Ztr); B=tf.transform(Zte)
    return IsotropicClassKDE(h).fit(A,ytr).predict_proba(B)


def _oof_logreg(X,y,dfdev,dataset,C):
    K=len(np.unique(y)); P=np.zeros((len(y),K),float)
    for tr,va in inner_splits(dfdev,dataset):
        mdl=make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C,solver='lbfgs',max_iter=5000,random_state=2026)
        ).fit(X[tr],y[tr])
        P[va]=mdl.predict_proba(X[va])
    return P


def _select_logreg(X,y,dfdev,dataset):
    rows=[]
    for C in CGRID:
        P=_oof_logreg(X,y,dfdev,dataset,C)
        rows.append({'C':C,'nll':nll(P,y)})
    best=min(r['nll'] for r in rows)
    cand=[r for r in rows if abs(r['nll']-best)<=TOL]
    chosen=sorted(cand,key=lambda r:r['C'])[0]
    return chosen,pd.DataFrame(rows)


def _fit_logreg(Xtr,ytr,Xte,C):
    mdl=make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C,solver='lbfgs',max_iter=5000,random_state=2026)
    ).fit(Xtr,ytr)
    return mdl.predict_proba(Xte)


def run_geometry_control(dataset, manifest, folds, Xsp, G, result_dir: Path, out_dir: Path,
                     eta=1e-6,gamma=1e-6,bootstrap_B=10000):
    """Secondary geometry control. Uses existing caches; does not retrain CNN or recompute DCT."""
    if dataset not in {'kth_tips2b','fmd','kylberg'}:
        raise RuntimeError('geometry_control is defined for the three primary datasets only')
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    selected=pd.read_csv(Path(result_dir)/'selected_configs.csv')
    oldpred=pd.read_csv(Path(result_dir)/'outer_predictions.csv')
    K=int(manifest.class_id.nunique())
    rows=[]; sel_rows=[]

    for fold in sorted(folds.outer_fold.unique()):
        fp=out_dir/f'fold_{int(fold)}.npz'
        sp=out_dir/f'fold_{int(fold)}_selected.json'
        if fp.exists() and sp.exists():
            d=np.load(fp); te=d['test_index']; P4=d['P_B4W']; P3=d['P_C3']
            info=json.loads(sp.read_text())
        else:
            te=np.flatnonzero(folds.outer_fold.values==fold)
            tr=np.flatnonzero(folds.outer_fold.values!=fold)
            ytr=manifest.class_id.values[tr]; yte=manifest.class_id.values[te]
            dfdev=manifest.iloc[tr].reset_index(drop=True)

            # B4W: exactly the B5 separate expert's selected R and h in the same outer fold.
            srow=selected[selected.outer_fold==fold].iloc[0]
            R=int(srow['spectral_separate.R']); h=float(srow['spectral_separate.h_f'])
            P4=_fit_b4w_matched(G[tr].astype(float),ytr,G[te].astype(float),R,h,eta,gamma)

            # C3: strong probabilistic linear baseline on frozen 512-D ResNet embeddings.
            c3,cand=_select_logreg(Xsp[tr].astype(float),ytr,dfdev,dataset)
            cand.to_csv(out_dir/f'fold_{int(fold)}_C3_candidates.csv',index=False)
            P3=_fit_logreg(Xsp[tr].astype(float),ytr,Xsp[te].astype(float),float(c3['C']))

            tmp=fp.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,test_index=te,y_true=yte,P_B4W=P4,P_C3=P3)
            tmp.replace(fp)
            info={'outer_fold':int(fold),'B4W_R':R,'B4W_h':h,'C3_C':float(c3['C']),'C3_inner_nll':float(c3['nll'])}
            atomic_json(sp,info)

        sel_rows.append(info)
        yte=manifest.class_id.values[te]
        for pos,idx in enumerate(te):
            rec={'image_id':manifest.image_id.iloc[idx],'class_id':int(yte[pos]),'outer_fold':int(fold)}
            for k in range(K):
                rec[f'p__B4W__{k}']=float(P4[pos,k]); rec[f'p__C3__{k}']=float(P3[pos,k])
            rows.append(rec)

    pred=pd.DataFrame(rows).sort_values('image_id').reset_index(drop=True)
    pred.to_csv(out_dir/'geometry_control_outer_predictions.csv',index=False)
    pd.DataFrame(sel_rows).to_csv(out_dir/'geometry_control_selected.csv',index=False)

    y=pred.class_id.to_numpy()
    P4=pred[_prob_cols(pred,'B4W',K)].to_numpy()
    P3=pred[_prob_cols(pred,'C3',K)].to_numpy()
    metrics=pd.DataFrame([
        {'dataset':dataset,'method':'B4W',**all_metrics(y,P=P4)},
        {'dataset':dataset,'method':'C3_LogReg',**all_metrics(y,P=P3)},
    ])
    metrics.to_csv(out_dir/'geometry_control_metrics.csv',index=False)

    # Align existing B5 predictions by image_id for paired mechanism inference.
    oo=oldpred.set_index('image_id').loc[pred.image_id].reset_index()
    PB5=oo[[f'p__B5__{k}' for k in range(K)]].to_numpy()
    meta=manifest.set_index('image_id').loc[pred.image_id].reset_index()
    clustered=dataset in {'kth_tips2b','kylberg'}
    ci=paired_bootstrap(meta.reset_index(drop=True),PB5,P4,B=bootstrap_B,seed=2026,clustered=clustered)
    m5=all_metrics(y,P=PB5); m4=all_metrics(y,P=P4)
    effect={
        'dataset':dataset,
        'comparison':'B5_minus_B4W',
        'delta_macro_f1':float(m5['macro_f1']-m4['macro_f1']),
        'delta_nll':float(m5['nll']-m4['nll']),
        **ci,
    }
    atomic_json(out_dir/'B5_minus_B4W_bootstrap.json',effect)

    # Zero-energy boundary audit from cached coordinate energy.
    # Caller writes actual S_E summary because it owns the npz path.
    return metrics,effect,pd.DataFrame(sel_rows)
