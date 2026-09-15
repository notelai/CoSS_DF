from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .datasets.splits import inner_splits
from .features.spectral import spectral_from_G, directional_spectral_from_G, helmert_basis
from .models.transforms import RegularizedWhitener
from .models.density import IsotropicClassKDE
from .models.fusion import nll, linear_pool, logop, optimize_lambda
from .metrics import all_metrics
from .analysis.bootstrap import paired_bootstrap
from .utils import atomic_json

CGRID=(.01,.1,1.,10.,100.)
HGRID=(.25,.5,.75,1.,1.5,2.)
RGRID=(4,6,8)
TOL=1e-12


def _prob_cols(method,K):
    return [f"p__{method}__{k}" for k in range(K)]


def _align_proba(model,P,K):
    out=np.zeros((len(P),K),float)
    for j,c in enumerate(model.classes_):
        out[:,int(c)]=P[:,j]
    return out


def _spatial_lr_fit(Xtr,ytr,Xte,C,K):
    mdl=make_pipeline(StandardScaler(),LogisticRegression(C=C,solver='lbfgs',max_iter=5000,random_state=2026))
    mdl.fit(Xtr,ytr)
    return _align_proba(mdl[-1],mdl.predict_proba(Xte),K)


def _spectral_coords(G,R,kind='ilr',eta=1e-6):
    _,P,Z=spectral_from_G(G,R,eta)
    if kind=='ilr':
        return Z
    if kind=='linear':
        return P@helmert_basis(R)
    raise ValueError(kind)


def _spectral_lr_fit(Gtr,ytr,Gte,R,C,K,kind='ilr',eta=1e-6,gamma=1e-6):
    Ztr=_spectral_coords(Gtr,R,kind,eta); Zte=_spectral_coords(Gte,R,kind,eta)
    wh=RegularizedWhitener(gamma).fit(Ztr)
    A=wh.transform(Ztr); B=wh.transform(Zte)
    mdl=LogisticRegression(C=C,solver='lbfgs',max_iter=5000,random_state=2026).fit(A,ytr)
    return _align_proba(mdl,mdl.predict_proba(B),K)


def _spatial_lr_oof(X,y,dfdev,dataset,C,K):
    P=np.zeros((len(y),K),float)
    for tr,va in inner_splits(dfdev,dataset):
        P[va]=_spatial_lr_fit(X[tr],y[tr],X[va],C,K)
    return P


def _spectral_lr_oof(G,y,dfdev,dataset,R,C,K,kind='ilr',eta=1e-6,gamma=1e-6):
    P=np.zeros((len(y),K),float)
    for tr,va in inner_splits(dfdev,dataset):
        P[va]=_spectral_lr_fit(G[tr],y[tr],G[va],R,C,K,kind,eta,gamma)
    return P


def _select_spatial_lr(X,y,dfdev,dataset,K):
    rows=[]; probs={}
    for C in CGRID:
        P=_spatial_lr_oof(X,y,dfdev,dataset,C,K)
        sc=nll(P,y); rows.append({'C':C,'score':sc}); probs[C]=P
    best=min(r['score'] for r in rows); cand=[r for r in rows if abs(r['score']-best)<=TOL]
    chosen=sorted(cand,key=lambda r:r['C'])[0]
    return chosen,pd.DataFrame(rows),probs[chosen['C']]


def _select_spectral_lr(G,y,dfdev,dataset,K,kind='ilr',eta=1e-6,gamma=1e-6):
    rows=[]; probs={}
    for R in RGRID:
        for C in CGRID:
            P=_spectral_lr_oof(G,y,dfdev,dataset,R,C,K,kind,eta,gamma)
            sc=nll(P,y); rows.append({'R':R,'C':C,'score':sc}); probs[(R,C)]=P
    best=min(r['score'] for r in rows); cand=[r for r in rows if abs(r['score']-best)<=TOL]
    chosen=sorted(cand,key=lambda r:(r['R'],r['C']))[0]
    return chosen,pd.DataFrame(rows),probs


def _select_matched_geometry_lr(G,y,dfdev,dataset,K,eta=1e-6,gamma=1e-6,ilr_cache=None):
    """Symmetric inner selection of a shared (R,C) for linear-vs-ILR logistic control.

    The selection criterion is the arithmetic mean of the two OOF NLL values, so
    neither representation is privileged by hyperparameter selection.
    """
    rows=[]; cache={}
    for R in RGRID:
        for C in CGRID:
            Pl=_spectral_lr_oof(G,y,dfdev,dataset,R,C,K,'linear',eta,gamma)
            Pi=(ilr_cache[(R,C)] if ilr_cache is not None and (R,C) in ilr_cache else _spectral_lr_oof(G,y,dfdev,dataset,R,C,K,'ilr',eta,gamma))
            nl=nll(Pl,y); ni=nll(Pi,y); sym=.5*(nl+ni)
            rows.append({'R':R,'C':C,'linear_nll':nl,'ilr_nll':ni,'symmetric_score':sym})
            cache[(R,C)]=(Pl,Pi)
    best=min(r['symmetric_score'] for r in rows); cand=[r for r in rows if abs(r['symmetric_score']-best)<=TOL]
    chosen=sorted(cand,key=lambda r:(r['R'],r['C']))[0]
    return chosen,pd.DataFrame(rows),cache[(chosen['R'],chosen['C'])]


def _directional_kde_fit(Gtr,ytr,Gte,R,h,K,sectors=2,eta=1e-6,gamma=1e-6):
    Ztr=directional_spectral_from_G(Gtr,R,sectors,eta)[2]
    Zte=directional_spectral_from_G(Gte,R,sectors,eta)[2]
    wh=RegularizedWhitener(gamma).fit(Ztr)
    A=wh.transform(Ztr); B=wh.transform(Zte)
    return IsotropicClassKDE(h).fit(A,ytr).predict_proba(B)


def _directional_kde_oof(G,y,dfdev,dataset,R,h,K,sectors=2,eta=1e-6,gamma=1e-6):
    P=np.zeros((len(y),K),float)
    for tr,va in inner_splits(dfdev,dataset):
        P[va]=_directional_kde_fit(G[tr],y[tr],G[va],R,h,K,sectors,eta,gamma)
    return P


def _select_directional_h(G,y,dfdev,dataset,R,K,sectors=2,eta=1e-6,gamma=1e-6):
    rows=[]; probs={}
    for h in HGRID:
        P=_directional_kde_oof(G,y,dfdev,dataset,R,h,K,sectors,eta,gamma)
        sc=nll(P,y); rows.append({'R':R,'sectors':sectors,'h':h,'score':sc,'neg_h':-h}); probs[h]=P
    best=min(r['score'] for r in rows); cand=[r for r in rows if abs(r['score']-best)<=TOL]
    chosen=sorted(cand,key=lambda r:r['neg_h'])[0]  # larger h under exact tie, matching paper contract
    return chosen,pd.DataFrame(rows),probs[chosen['h']]


def _diagnostics(y,Ps,Pf,Pfused,name):
    ys=Ps.argmax(1); yf=Pf.argmax(1); yu=Pfused.argmax(1)
    sw=(ys!=y); sc=(ys==y)
    eps=1e-300
    gain=np.log(np.maximum(Pfused[np.arange(len(y)),y],eps))-np.log(np.maximum(Ps[np.arange(len(y)),y],eps))
    return {
        'fusion':name,
        'n':int(len(y)),
        'expert_disagreement':float(np.mean(ys!=yf)),
        'spectral_correction_rate_given_spatial_wrong':float(np.mean(yf[sw]==y[sw])) if sw.any() else np.nan,
        'spectral_harm_rate_given_spatial_correct':float(np.mean(yf[sc]!=y[sc])) if sc.any() else np.nan,
        'fusion_correction_rate_given_spatial_wrong':float(np.mean(yu[sw]==y[sw])) if sw.any() else np.nan,
        'fusion_harm_rate_given_spatial_correct':float(np.mean(yu[sc]!=y[sc])) if sc.any() else np.nan,
        'mean_logloss_gain_vs_spatial':float(np.mean(gain)),
        'mean_gain_spatial_wrong':float(np.mean(gain[sw])) if sw.any() else np.nan,
        'mean_gain_spatial_correct':float(np.mean(gain[sc])) if sc.any() else np.nan,
        'fraction_positive_logloss_gain':float(np.mean(gain>0)),
    }


def _safe_npz(path,**kw):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**kw); tmp.replace(path)


def run_directional_fusion_controls(dataset,manifest,folds,Xsp,G,result_dir:Path,out_dir:Path,
                     eta=1e-6,gamma=1e-6,sectors=2,bootstrap_B=10000):
    if dataset not in {'kth_tips2b','fmd','kylberg'}:
        raise RuntimeError('directional_fusion_controls is defined for the three primary datasets only')
    if sectors!=2:
        raise RuntimeError('The directional sensitivity analysis is fixed to sectors=2')
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    selected=pd.read_csv(Path(result_dir)/'selected_configs.csv')
    oldpred=pd.read_csv(Path(result_dir)/'outer_predictions.csv')
    K=int(manifest.class_id.nunique())
    rows=[]; selections=[]

    for fold in sorted(folds.outer_fold.unique()):
        fp=out_dir/f'fold_{int(fold)}.npz'; sp=out_dir/f'fold_{int(fold)}_selected.json'
        if fp.exists() and sp.exists():
            d=np.load(fp); te=d['test_index']; payload={k:d[k] for k in d.files if k not in {'test_index','y_true'}}
            info=json.loads(sp.read_text())
        else:
            te=np.flatnonzero(folds.outer_fold.values==fold); tr=np.flatnonzero(folds.outer_fold.values!=fold)
            ytr=manifest.class_id.values[tr]; yte=manifest.class_id.values[te]
            dfdev=manifest.iloc[tr].reset_index(drop=True)
            Xtr=Xsp[tr].astype(float); Xte=Xsp[te].astype(float); Gtr=G[tr].astype(float); Gte=G[te].astype(float)
            srow=selected[selected.outer_fold==fold].iloc[0]

            # E2: directional radial-sector spectral KDE.
            # Primary matched control uses the original radial B5 R and h.
            Rrad=int(srow['spectral_separate.R']); hrad=float(srow['spectral_separate.h_f'])
            Pdir_matched=_directional_kde_fit(Gtr,ytr,Gte,Rrad,hrad,K,sectors,eta,gamma)
            dsel,dtab,_=_select_directional_h(Gtr,ytr,dfdev,dataset,Rrad,K,sectors,eta,gamma)
            dtab.to_csv(out_dir/f'fold_{int(fold)}_directional_h_candidates.csv',index=False)
            Pdir_tuned=_directional_kde_fit(Gtr,ytr,Gte,Rrad,float(dsel['h']),K,sectors,eta,gamma)

            # E3: strong spatial and spectral logistic experts.
            ssel,stab,Ps_oof=_select_spatial_lr(Xtr,ytr,dfdev,dataset,K)
            fsel,ftab,fprob_cache=_select_spectral_lr(Gtr,ytr,dfdev,dataset,K,'ilr',eta,gamma)
            Pf_oof=fprob_cache[(int(fsel['R']),float(fsel['C']))]
            stab.to_csv(out_dir/f'fold_{int(fold)}_C3s_candidates.csv',index=False)
            ftab.to_csv(out_dir/f'fold_{int(fold)}_C3f_candidates.csv',index=False)
            lam_lin,lin_nll=optimize_lambda(Ps_oof,Pf_oof,ytr,'linear')
            lam_log,log_nll=optimize_lambda(Ps_oof,Pf_oof,ytr,'logop')
            Ps=_spatial_lr_fit(Xtr,ytr,Xte,float(ssel['C']),K)
            Pf=_spectral_lr_fit(Gtr,ytr,Gte,int(fsel['R']),float(fsel['C']),K,'ilr',eta,gamma)
            Pstrong_lin=linear_pool(Ps,Pf,lam_lin); Pstrong_log=logop(Ps,Pf,lam_log)

            # E3d: classifier-independent geometry control with symmetric shared selection.
            gsel,gtab,_=_select_matched_geometry_lr(Gtr,ytr,dfdev,dataset,K,eta,gamma,ilr_cache=fprob_cache)
            gtab.to_csv(out_dir/f'fold_{int(fold)}_matched_geometry_lr_candidates.csv',index=False)
            Rg=int(gsel['R']); Cg=float(gsel['C'])
            Pgeom_linear=_spectral_lr_fit(Gtr,ytr,Gte,Rg,Cg,K,'linear',eta,gamma)
            Pgeom_ilr=_spectral_lr_fit(Gtr,ytr,Gte,Rg,Cg,K,'ilr',eta,gamma)

            payload={
                'P_DIR_MATCHED':Pdir_matched,'P_DIR_TUNED':Pdir_tuned,
                'P_C3S':Ps,'P_C3F':Pf,'P_C3_LINEAR_FUSION':Pstrong_lin,'P_C3_LOGOP_FUSION':Pstrong_log,
                'P_LR_LINEAR_GEOM':Pgeom_linear,'P_LR_ILR_GEOM':Pgeom_ilr,
            }
            _safe_npz(fp,test_index=te,y_true=yte,**payload)
            info={
                'outer_fold':int(fold),'directional_R':Rrad,'directional_h_matched':hrad,'directional_h_tuned':float(dsel['h']),
                'C3s_C':float(ssel['C']),'C3f_R':int(fsel['R']),'C3f_C':float(fsel['C']),
                'C3_linear_lambda':float(lam_lin),'C3_linear_inner_nll':float(lin_nll),
                'C3_logop_lambda':float(lam_log),'C3_logop_inner_nll':float(log_nll),
                'matched_geometry_R':Rg,'matched_geometry_C':Cg,
                'matched_geometry_linear_inner_nll':float(gsel['linear_nll']),
                'matched_geometry_ilr_inner_nll':float(gsel['ilr_nll']),
                'matched_geometry_symmetric_score':float(gsel['symmetric_score']),
                'directional_sectors':int(sectors),
            }
            atomic_json(sp,info)

        selections.append(info); yte=manifest.class_id.values[te]
        for pos,idx in enumerate(te):
            rec={'image_id':manifest.image_id.iloc[idx],'class_id':int(yte[pos]),'outer_fold':int(fold)}
            for method,P in payload.items():
                name=method[2:] if method.startswith('P_') else method
                for k in range(K): rec[f'p__{name}__{k}']=float(P[pos,k])
            rows.append(rec)

    pred=pd.DataFrame(rows).sort_values('image_id').reset_index(drop=True)
    pred.to_csv(out_dir/'directional_fusion_controls_outer_predictions.csv',index=False)
    pd.DataFrame(selections).to_csv(out_dir/'directional_fusion_controls_selected.csv',index=False)
    y=pred.class_id.to_numpy()

    method_names=['DIR_MATCHED','DIR_TUNED','C3S','C3F','C3_LINEAR_FUSION','C3_LOGOP_FUSION','LR_LINEAR_GEOM','LR_ILR_GEOM']
    metrics=[]
    for method in method_names:
        P=pred[_prob_cols(method,K)].to_numpy()
        metrics.append({'dataset':dataset,'method':method,**all_metrics(y,P=P)})
    met=pd.DataFrame(metrics); met.to_csv(out_dir/'directional_fusion_controls_metrics.csv',index=False)

    # Existing radial B5 aligned to the secondary-analysis rows.
    oo=oldpred.set_index('image_id').loc[pred.image_id].reset_index()
    PB5=oo[[f'p__B5__{k}' for k in range(K)]].to_numpy()
    meta=manifest.set_index('image_id').loc[pred.image_id].reset_index()
    clustered=dataset in {'kth_tips2b','kylberg'}

    effects=[]
    comparisons=[
        ('DIR_MATCHED_minus_B5','DIR_MATCHED',PB5),
        ('DIR_TUNED_minus_B5','DIR_TUNED',PB5),
        ('C3_LINEAR_FUSION_minus_C3S','C3_LINEAR_FUSION',pred[_prob_cols('C3S',K)].to_numpy()),
        ('C3_LOGOP_FUSION_minus_C3S','C3_LOGOP_FUSION',pred[_prob_cols('C3S',K)].to_numpy()),
        ('LR_ILR_GEOM_minus_LR_LINEAR_GEOM','LR_ILR_GEOM',pred[_prob_cols('LR_LINEAR_GEOM',K)].to_numpy()),
    ]
    for cname,amethod,PB in comparisons:
        PA=pred[_prob_cols(amethod,K)].to_numpy()
        ma=all_metrics(y,P=PA); mb=all_metrics(y,P=PB)
        ci=paired_bootstrap(meta.reset_index(drop=True),PA,PB,B=bootstrap_B,seed=2026,clustered=clustered)
        eff={'dataset':dataset,'comparison':cname,'delta_macro_f1':float(ma['macro_f1']-mb['macro_f1']),
             'delta_nll':float(ma['nll']-mb['nll']),**ci}
        effects.append(eff)
    pd.DataFrame(effects).to_csv(out_dir/'directional_fusion_controls_effects.csv',index=False)
    atomic_json(out_dir/'directional_fusion_controls_effects.json',effects)

    # E4 diagnostics for strong-expert fusion.
    Ps=pred[_prob_cols('C3S',K)].to_numpy(); Pf=pred[_prob_cols('C3F',K)].to_numpy()
    Pl=pred[_prob_cols('C3_LINEAR_FUSION',K)].to_numpy(); Pg=pred[_prob_cols('C3_LOGOP_FUSION',K)].to_numpy()
    diag=pd.DataFrame([_diagnostics(y,Ps,Pf,Pl,'C3_LINEAR_FUSION'),_diagnostics(y,Ps,Pf,Pg,'C3_LOGOP_FUSION')])
    diag.insert(0,'dataset',dataset); diag.to_csv(out_dir/'directional_fusion_controls_synergy_diagnostics.csv',index=False)

    return met,pd.DataFrame(effects),diag,pd.DataFrame(selections)
