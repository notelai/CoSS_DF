from __future__ import annotations
import json, itertools, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .datasets.splits import inner_splits
from .features.spectral import spectral_from_G
from .models.transforms import Standardizer, SpatialPCAWhitener, RegularizedWhitener
from .models.density import IsotropicClassKDE, GaussianClassDensity
from .models.fusion import nll, logop, linear_pool, optimize_lambda
from .metrics import all_metrics
from .utils import atomic_json

HGRID=[.25,.5,.75,1.,1.5,2.]
DSGRID=[4,8,12,16]
RGRID=[4,6,8]
TOL=1e-12

def _safe_npz(path, **kw):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp.npz"); np.savez_compressed(tmp,**kw); tmp.replace(path)

def _load_prob(path): return np.load(path)["P"]

def _tie_min(rows, fields):
    best=min(r["score"] for r in rows); cand=[r for r in rows if abs(r["score"]-best)<=TOL]
    return sorted(cand,key=lambda r:tuple(r[f] for f in fields))[0]

def _oof_spatial(X,y,dfdev,dataset,cache):
    rows=[]; probs={}; K=len(np.unique(y))
    for d,h in itertools.product(DSGRID,HGRID):
        key=f"d{d}_h{h:g}"; pth=cache/f"spatial_{key}.npz"
        if pth.exists(): P=_load_prob(pth)
        else:
            P=np.zeros((len(y),K),float)
            for tr,va in inner_splits(dfdev,dataset):
                tf=SpatialPCAWhitener(d).fit(X[tr]); A=tf.transform(X[tr]); B=tf.transform(X[va]); kde=IsotropicClassKDE(h).fit(A,y[tr]); P[va]=kde.predict_proba(B)
            _safe_npz(pth,P=P)
        score=nll(P,y); rows.append({"d_s":d,"h_s":h,"score":score,"neg_h":-h}); probs[(d,h)]=P
    return rows,probs

def _oof_spectral(G,y,dfdev,dataset,cache,eta=1e-6,gamma=1e-6):
    rows=[]; probs={}; K=len(np.unique(y)); Zs={R:spectral_from_G(G,R,eta)[2] for R in RGRID}
    for R,h in itertools.product(RGRID,HGRID):
        pth=cache/f"spectral_R{R}_h{h:g}.npz"
        if pth.exists(): P=_load_prob(pth)
        else:
            Z=Zs[R]; P=np.zeros((len(y),K),float)
            for tr,va in inner_splits(dfdev,dataset):
                tf=RegularizedWhitener(gamma).fit(Z[tr]); A=tf.transform(Z[tr]); B=tf.transform(Z[va]); kde=IsotropicClassKDE(h).fit(A,y[tr]); P[va]=kde.predict_proba(B)
            _safe_npz(pth,P=P)
        score=nll(P,y); rows.append({"R":R,"h_f":h,"score":score,"neg_h":-h}); probs[(R,h)]=P
    return rows,probs,Zs

def _oof_generic_density(X_by_param,y,dfdev,dataset,cache,prefix,transform_kind="standard"):
    rows=[]; probs={}; K=len(np.unique(y))
    for param,X in X_by_param.items():
        for h in HGRID:
            key=f"{param}_h{h:g}"; pth=cache/f"{prefix}_{key}.npz"
            if pth.exists(): P=_load_prob(pth)
            else:
                P=np.zeros((len(y),K),float)
                for tr,va in inner_splits(dfdev,dataset):
                    tf=Standardizer().fit(X[tr]); A=tf.transform(X[tr]); B=tf.transform(X[va]); P[va]=IsotropicClassKDE(h).fit(A,y[tr]).predict_proba(B)
                _safe_npz(pth,P=P)
            rows.append({"param":param,"h":h,"score":nll(P,y),"neg_h":-h}); probs[(param,h)]=P
    return rows,probs

def _select_joint(srows,sprobs,frows,fprobs,y):
    rows=[]
    for sr in srows:
        for fr in frows:
            Ps=sprobs[(sr["d_s"],sr["h_s"])]; Pf=fprobs[(fr["R"],fr["h_f"])]
            lam,sc=optimize_lambda(Ps,Pf,y,"logop")
            rows.append({"d_s":sr["d_s"],"h_s":sr["h_s"],"R":fr["R"],"h_f":fr["h_f"],"lambda":lam,"score":sc})
    best=min(r["score"] for r in rows); cand=[r for r in rows if abs(r["score"]-best)<=TOL]
    chosen=sorted(cand,key=lambda r:(r["d_s"],r["R"],-r["h_s"],-r["h_f"]))[0]
    return chosen,pd.DataFrame(rows)

def _fit_spatial(Xtr,ytr,Xte,d,h):
    tf=SpatialPCAWhitener(d).fit(Xtr); A=tf.transform(Xtr); B=tf.transform(Xte); return IsotropicClassKDE(h).fit(A,ytr).predict_proba(B),tf,A,B

def _fit_spectral(Gtr,ytr,Gte,R,h,eta=1e-6,gamma=1e-6):
    Ztr=spectral_from_G(Gtr,R,eta)[2]; Zte=spectral_from_G(Gte,R,eta)[2]; tf=RegularizedWhitener(gamma).fit(Ztr); A=tf.transform(Ztr); B=tf.transform(Zte); return IsotropicClassKDE(h).fit(A,ytr).predict_proba(B),tf,A,B

def _fit_generic_kde(Xtr,ytr,Xte,h):
    tf=Standardizer().fit(Xtr); A=tf.transform(Xtr); B=tf.transform(Xte); return IsotropicClassKDE(h).fit(A,ytr).predict_proba(B)

def _select_svm(X,y,dfdev,dataset,kernels=("linear",),Cgrid=(.01,.1,1,10,100),gammas=(2**-4,2**-2,1,2**2,2**4)):
    rows=[]
    for ker in kernels:
        gs=[None] if ker=="linear" else gammas
        for C in Cgrid:
            for gscl in gs:
                pred=np.zeros(len(y),int)
                for tr,va in inner_splits(dfdev,dataset):
                    gamma="scale" if gscl is None else gscl/X.shape[1]
                    mdl=make_pipeline(StandardScaler(),SVC(C=C,kernel=ker,gamma=gamma)).fit(X[tr],y[tr]); pred[va]=mdl.predict(X[va])
                sc=f1_score(y,pred,average="macro"); rows.append({"kernel":ker,"C":C,"gamma_scale":gscl,"score":sc})
    best=max(r["score"] for r in rows); cand=[r for r in rows if abs(r["score"]-best)<=TOL]
    chosen=sorted(cand,key=lambda r:(0 if r["kernel"]=="linear" else 1,r["C"],float("-inf") if r["gamma_scale"] is None else r["gamma_scale"]))[0]
    return chosen,pd.DataFrame(rows)

def run_outer_fold(dataset,manifest,folds,Xsp,G,lbp_features,outer_fold,out_dir,eta=1e-6,gamma=1e-6):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); final=out/f"outer_fold_{outer_fold}.npz"; selected_path=out/f"selected_fold_{outer_fold}.json"
    if final.exists() and selected_path.exists(): return final,selected_path
    te=np.flatnonzero(folds.outer_fold.values==outer_fold); tr=np.flatnonzero(folds.outer_fold.values!=outer_fold)
    dfdev=manifest.iloc[tr].reset_index(drop=True); ytr=manifest.class_id.values[tr]; yte=manifest.class_id.values[te]
    Xtr=Xsp[tr].astype(float); Xte=Xsp[te].astype(float); Gtr=G[tr].astype(float); Gte=G[te].astype(float)
    cache=out/f"inner_fold_{outer_fold}"; cache.mkdir(parents=True,exist_ok=True)

    srows,sprobs=_oof_spatial(Xtr,ytr,dfdev,dataset,cache)
    frows,fprobs,Zs=_oof_spectral(Gtr,ytr,dfdev,dataset,cache,eta,gamma)
    ssep=_tie_min(srows,["d_s","neg_h"]); fsep=_tie_min(frows,["R","neg_h"])
    joint,joint_table=_select_joint(srows,sprobs,frows,fprobs,ytr); joint_table.to_csv(cache/"joint_candidates.csv",index=False)

    # Separate experts
    P_B1,tf_s,A_s,B_s=_fit_spatial(Xtr,ytr,Xte,ssep["d_s"],ssep["h_s"])
    P_B5,tf_f,A_f,B_f=_fit_spectral(Gtr,ytr,Gte,fsep["R"],fsep["h_f"],eta,gamma)
    P_B1G=GaussianClassDensity().fit(A_s,ytr).predict_proba(B_s)
    P_B5G=GaussianClassDensity().fit(A_f,ytr).predict_proba(B_f)

    # Controlled fusion: lambda from OOF selected separate expert probs.
    Ps_oof=sprobs[(ssep["d_s"],ssep["h_s"])]; Pf_oof=fprobs[(fsep["R"],fsep["h_f"])]
    lam_lin,_=optimize_lambda(Ps_oof,Pf_oof,ytr,"linear"); lam_log,_=optimize_lambda(Ps_oof,Pf_oof,ytr,"logop")
    P_F1=linear_pool(P_B1,P_B5,.5); P_F2=linear_pool(P_B1,P_B5,lam_lin); P_F3=logop(P_B1,P_B5,.5); P_F4=logop(P_B1,P_B5,lam_log)

    # Joint primary
    P_js,_,_,_=_fit_spatial(Xtr,ytr,Xte,joint["d_s"],joint["h_s"])
    P_jf,_,_,_=_fit_spectral(Gtr,ytr,Gte,joint["R"],joint["h_f"],eta,gamma)
    P_CoSS=logop(P_js,P_jf,joint["lambda"])

    # B2: 63 coordinate energy values, fixed representation; B3/B4 choose R and h independently by OOF NLL.
    mask=np.ones((8,8),bool); mask[0,0]=False; B2tr=Gtr[:,mask]
    r2,p2=_oof_generic_density({"fixed":B2tr},ytr,dfdev,dataset,cache,"b2"); b2=_tie_min(r2,["neg_h"])
    P_B2=_fit_generic_kde(B2tr,ytr,Gte[:,mask],b2["h"])
    Etr={f"R{R}":spectral_from_G(Gtr,R,eta)[0] for R in RGRID}; Ete={f"R{R}":spectral_from_G(Gte,R,eta)[0] for R in RGRID}
    Ptr={f"R{R}":spectral_from_G(Gtr,R,eta)[1] for R in RGRID}; Pte={f"R{R}":spectral_from_G(Gte,R,eta)[1] for R in RGRID}
    r3,p3=_oof_generic_density(Etr,ytr,dfdev,dataset,cache,"b3"); b3=_tie_min(r3,["param","neg_h"])
    r4,p4=_oof_generic_density(Ptr,ytr,dfdev,dataset,cache,"b4"); b4=_tie_min(r4,["param","neg_h"])
    P_B3=_fit_generic_kde(Etr[b3["param"]],ytr,Ete[b3["param"]],b3["h"])
    P_B4=_fit_generic_kde(Ptr[b4["param"]],ytr,Pte[b4["param"]],b4["h"])

    # C2 ResNet18 SVM
    c2,c2tab=_select_svm(Xtr,ytr,dfdev,dataset,kernels=("linear",)); c2tab.to_csv(cache/"c2_candidates.csv",index=False)
    mdl=make_pipeline(StandardScaler(),SVC(C=c2["C"],kernel="linear")).fit(Xtr,ytr); pred_C2=mdl.predict(Xte)

    # C1 LBP + SVM, candidates across three frozen LBP specs.
    c1rows=[]; c1best=None
    for spec,Xall in lbp_features.items():
        sel,tab=_select_svm(Xall[tr],ytr,dfdev,dataset,kernels=("linear","rbf")); sel={**sel,"spec":spec}; c1rows.append(sel)
    best=max(r["score"] for r in c1rows); cc=[r for r in c1rows if abs(r["score"]-best)<=TOL]
    c1best=sorted(cc,key=lambda r:(r["spec"],0 if r["kernel"]=="linear" else 1,r["C"],-1 if r["gamma_scale"] is None else r["gamma_scale"]))[0]
    Xc=lbp_features[c1best["spec"]]; gam="scale" if c1best["gamma_scale"] is None else c1best["gamma_scale"]/Xc.shape[1]
    mdl1=make_pipeline(StandardScaler(),SVC(C=c1best["C"],kernel=c1best["kernel"],gamma=gam)).fit(Xc[tr],ytr); pred_C1=mdl1.predict(Xc[te])

    methods={"B1":P_B1,"B1G":P_B1G,"B2":P_B2,"B3":P_B3,"B4":P_B4,"B5":P_B5,"B5G":P_B5G,
             "F1_linear_fixed":P_F1,"F2_linear_opt":P_F2,"F3_logop_fixed":P_F3,"F4_logop_opt_controlled":P_F4,"CoSS_DF":P_CoSS}
    payload={"test_index":te,"y_true":yte,"pred_C1":pred_C1,"pred_C2":pred_C2}
    for name,P in methods.items(): payload[f"P__{name}"]=P
    _safe_npz(final,**payload)
    selected={"outer_fold":outer_fold,"spatial_separate":ssep,"spectral_separate":fsep,"joint":joint,"lambda_linear_controlled":lam_lin,"lambda_logop_controlled":lam_log,
              "B2":b2,"B3":b3,"B4":b4,"C1":c1best,"C2":c2}
    atomic_json(selected_path,selected)
    return final,selected_path

def aggregate_outer(dataset,manifest,folds,fold_dir,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); rows=[]; sels=[]; K=manifest.class_id.nunique()
    prob_methods=None
    for fp in sorted(Path(fold_dir).glob("outer_fold_*.npz")):
        d=np.load(fp); te=d["test_index"]; y=d["y_true"]; fold=int(fp.stem.split("_")[-1])
        if prob_methods is None: prob_methods=[k[3:] for k in d.files if k.startswith("P__")]
        for pos,idx in enumerate(te):
            base={"image_id":manifest.image_id.iloc[idx],"class_id":int(y[pos]),"class_name":manifest.class_name.iloc[idx],"group_id":manifest.group_id.iloc[idx],"source_id":manifest.source_id.iloc[idx],"outer_fold":fold,
                  "scale":manifest.scale.iloc[idx],"pose":manifest.pose.iloc[idx],"illumination":manifest.illumination.iloc[idx],
                  "pred_C1":int(d["pred_C1"][pos]),"pred_C2":int(d["pred_C2"][pos])}
            for m in prob_methods:
                P=d[f"P__{m}"][pos]; base[f"pred__{m}"]=int(P.argmax())
                for k,v in enumerate(P): base[f"p__{m}__{k}"]=float(v)
            rows.append(base)
    pred=pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True); pred.to_csv(out/"outer_predictions.csv",index=False)
    metrics=[]; y=pred.class_id.values
    metrics.append({"method":"C1","dataset":dataset,**all_metrics(y,pred=pred.pred_C1.values)})
    metrics.append({"method":"C2","dataset":dataset,**all_metrics(y,pred=pred.pred_C2.values)})
    for m in prob_methods:
        cols=[f"p__{m}__{k}" for k in range(K)]; P=pred[cols].values; metrics.append({"method":m,"dataset":dataset,**all_metrics(y,P=P)})
    mdf=pd.DataFrame(metrics); mdf.to_csv(out/"metrics.csv",index=False)
    for sp in sorted(Path(fold_dir).glob("selected_fold_*.json")): sels.append(json.loads(sp.read_text()))
    sdf=pd.json_normalize(sels); sdf.to_csv(out/"selected_configs.csv",index=False)
    return pred,mdf,sdf
