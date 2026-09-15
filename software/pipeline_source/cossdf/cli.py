from __future__ import annotations
import argparse, json, os, platform, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from .config import get_paths, load_frozen, load_datasets
from .checkpoints import CheckpointStore
from .datasets.download import download_dataset, extract_archive
from .datasets.audit import run_audit, assign_exact_duplicate_components, validate_exact_duplicates_do_not_cross_folds
from .datasets.splits import make_outer_folds, validate_no_group_leak
from .staging import stage_dataset_to_ssd
from .features.spatial import extract_spatial_chunked
from .features.spectral import extract_coordinate_energy_chunked
from .features.lbp import extract_lbp_chunked
from .features.alternative_spectral import extract_handcrafted_energy_chunked, clr_ilr_sanity
from .features.modern import extract_backbone_chunked
from .runner import run_outer_fold, aggregate_outer
from .analysis.bootstrap import paired_bootstrap
from .analysis.plots import reliability_plot, effect_size_plot
from .analysis.robustness import affine_verification_chunked, condition_metrics, variant_spectral_outer
from .metrics import all_metrics
from .geometry_control import run_geometry_control
from .directional_fusion_controls import run_directional_fusion_controls
from .additional_validation import run_additional_validation
from .utils import seed_everything, atomic_json, stable_hash


def _env_snapshot(path):
    import numpy, pandas, scipy, sklearn, PIL
    try:
        import torch, torchvision
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        tv=torchvision.__version__; th=torch.__version__
    except Exception: gpu=None; tv=th=None
    atomic_json(path,{"python":sys.version,"platform":platform.platform(),"numpy":numpy.__version__,"pandas":pandas.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"Pillow":PIL.__version__,"torch":th,"torchvision":tv,"gpu":gpu})

def _record_runtime(paths,dataset,stage,start):
    import psutil
    p=paths.results/dataset/"paper_v1_0"/"timing_memory.json"; old={}
    if p.exists(): old=json.loads(p.read_text())
    rec={"seconds":time.perf_counter()-start,"rss_gb":psutil.Process().memory_info().rss/1024**3}
    try:
        import torch
        rec["gpu_peak_gb"]=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0.0
    except Exception: rec["gpu_peak_gb"]=None
    old[stage]=rec; atomic_json(p,old)

def _spec(dataset): return load_datasets()[dataset]

def cmd_prepare(args):
    paths=get_paths(args.drive_root,args.local_root); spec=_spec(args.dataset); ck=CheckpointStore(paths.checkpoints,args.dataset)
    archive_dir=paths.datasets_drive/"archives"; extracted=paths.datasets_drive/"extracted"/args.dataset
    if not (extracted/".extract_complete").exists():
        arc=download_dataset(args.dataset,spec,archive_dir,args.source_url); extract_archive(arc,extracted)
    ck.mark_done("00_download_extract",artifacts=[str(extracted)])
    print(f"Drive dataset ready: {extracted}")

def _stage(args):
    paths=get_paths(args.drive_root,args.local_root); src=paths.datasets_drive/"extracted"/args.dataset; dst=paths.datasets_local/args.dataset
    return stage_dataset_to_ssd(src,dst,force=args.force_stage)

def cmd_audit(args):
    paths=get_paths(args.drive_root,args.local_root); spec=_spec(args.dataset); ck=CheckpointStore(paths.checkpoints,args.dataset); local=_stage(args)
    out=ck.artifact_dir("audit"); sig=ck.signature(dataset=args.dataset,stage="audit",expected=spec.get("expected_images"))
    if ck.is_done("01_audit",sig) and (out/"manifest.csv").exists(): print("[resume] audit checkpoint valid"); return
    df,summary=run_audit(args.dataset,local,out,spec.get("expected_images"),spec.get("expected_classes"),strict=not args.allow_audit_mismatch)
    df=assign_exact_duplicate_components(df); df.to_csv(out/"manifest.csv",index=False)
    folds=make_outer_folds(df,args.dataset,2026); validate_no_group_leak(folds); validate_exact_duplicates_do_not_cross_folds(df,folds); folds.to_csv(out/"fold_assignments.csv",index=False)
    ck.mark_done("01_audit",sig,[str(out/"manifest.csv"),str(out/"fold_assignments.csv")],summary)

def _load_manifest(paths,dataset):
    out=paths.checkpoints/dataset/"audit"; m=pd.read_csv(out/"manifest.csv"); f=pd.read_csv(out/"fold_assignments.csv")
    # pandas turns empty strings into NaN; keep group/source semantics, and rebuild local paths if needed.
    local=paths.datasets_local/dataset
    if not all(Path(p).exists() for p in m.filepath.head(min(10,len(m)))):
        raise RuntimeError("Local SSD dataset missing. Run stage/audit again in this session.")
    return m,f

def cmd_features(args):
    t0=time.perf_counter()
    paths=get_paths(args.drive_root,args.local_root); ck=CheckpointStore(paths.checkpoints,args.dataset); _stage(args)
    m,f=_load_manifest(paths,args.dataset); out=ck.artifact_dir("features")
    sig=ck.signature(stage="features",n=len(m),preprocess=224,backbone="resnet18_imagenet1k_v1",block=8)
    if not (out/"spatial"/"spatial_embeddings.npz").exists(): extract_spatial_chunked(m,out/"spatial",batch_size=args.batch_size,chunk_size=args.feature_chunk)
    if not (out/"spectral"/"coordinate_energy_o0_0.npz").exists(): extract_coordinate_energy_chunked(m,out/"spectral",chunk_size=args.spectral_chunk,m=8,offset=(0,0))
    if not args.skip_lbp: extract_lbp_chunked(m,out/"lbp",chunk_size=args.spectral_chunk)
    ck.mark_done("02_features",sig,[str(out)]); _record_runtime(paths,args.dataset,"features",t0)

def _load_features(paths,dataset,need_lbp=True):
    root=paths.checkpoints/dataset/"features"
    sd=np.load(root/"spatial"/"spatial_embeddings.npz"); X=sd["X"].astype(np.float64)
    gd=np.load(root/"spectral"/"coordinate_energy_o0_0.npz"); G=gd["G"].astype(np.float64)
    lbp={}
    if need_lbp:
        for P,R in [(8,1),(16,2),(24,3)]: lbp[f"P{P}_R{R}"]=np.load(root/"lbp"/f"P{P}_R{R}"/"features.npz")["X"].astype(np.float64)
    return X,G,lbp

def cmd_run(args):
    t0=time.perf_counter()
    paths=get_paths(args.drive_root,args.local_root); ck=CheckpointStore(paths.checkpoints,args.dataset); _stage(args); m,folds=_load_manifest(paths,args.dataset); X,G,lbp=_load_features(paths,args.dataset,need_lbp=not args.skip_lbp)
    if args.skip_lbp: raise RuntimeError("Paper-mode run requires C1 LBP features. Run features without --skip-lbp.")
    out=ck.artifact_dir("modeling")
    for fold in sorted(folds.outer_fold.unique()):
        print(f"=== OUTER FOLD {fold} ===")
        run_outer_fold(args.dataset,m,folds,X,G,lbp,int(fold),out)
    result_dir=paths.results/args.dataset/"paper_v1_0"; pred,met,sel=aggregate_outer(args.dataset,m,folds,out,result_dir)
    _env_snapshot(result_dir/"environment_snapshot.json")
    (result_dir/"paper_tables").mkdir(parents=True,exist_ok=True)
    met.to_csv(result_dir/"paper_tables"/f"{args.dataset}_metrics.csv",index=False); sel.to_csv(result_dir/"paper_tables"/f"{args.dataset}_selected_configs.csv",index=False)
    atomic_json(result_dir/"paper_results.json",{"dataset":args.dataset,"metrics":met.to_dict("records"),"selected_configs":sel.to_dict("records")})
    ck.mark_done("03_modeling",artifacts=[str(result_dir)]); _record_runtime(paths,args.dataset,"modeling",t0)
    print(met.to_string(index=False))

def _prob(pred,method,K): return pred[[f"p__{method}__{k}" for k in range(K)]].values

def cmd_analyze(args):
    t0=time.perf_counter()
    paths=get_paths(args.drive_root,args.local_root); ck=CheckpointStore(paths.checkpoints,args.dataset); _stage(args); m,folds=_load_manifest(paths,args.dataset); X,G,_=_load_features(paths,args.dataset,need_lbp=False)
    result_dir=paths.results/args.dataset/"paper_v1_0"; pred=pd.read_csv(result_dir/"outer_predictions.csv"); met=pd.read_csv(result_dir/"metrics.csv"); sel=result_dir/"selected_configs.csv"; K=m.class_id.nunique()
    # align manifest to prediction image IDs
    meta=m.set_index("image_id").loc[pred.image_id].reset_index(); PA=_prob(pred,"CoSS_DF",K); PB=_prob(pred,"B1",K)
    clustered=args.dataset in {"kth_tips2b","kylberg"}; boot=paired_bootstrap(meta.reset_index(drop=True),PA,PB,B=10000,seed=2026,clustered=clustered)
    a=met.set_index("method"); effect={"dataset":args.dataset,"delta_macro_f1":float(a.loc["CoSS_DF","macro_f1"]-a.loc["B1","macro_f1"]),"delta_nll":float(a.loc["CoSS_DF","nll"]-a.loc["B1","nll"]),**boot}
    atomic_json(result_dir/"bootstrap_effect.json",effect)
    reliability_plot({"B1":PB,"B5":_prob(pred,"B5",K),"CoSS-DF":PA},pred.class_id.values,result_dir/"figures"/"reliability.pdf")
    if args.dataset=="kth_tips2b":
        condition_metrics(pred,"CoSS_DF").to_csv(result_dir/"paper_tables"/"kth_conditions.csv",index=False)
    # Exact theorem implementation verification is checkpointed separately; it is CPU-heavy but resumable at analysis stage.
    av=result_dir/"paper_tables"/"affine_verification.csv"
    if not av.exists(): affine_verification_chunked(m,result_dir/"paper_tables",chunk_size=64)
    avdf=pd.read_csv(av); avsum=avdf.groupby(["a","b"],as_index=False).agg(max_delta_p=("delta_p_inf","max"),max_delta_ilr=("delta_ilr_l2","max"),mean_delta_E=("delta_E_rel","mean")); avsum.to_csv(result_dir/"paper_tables"/"affine_verification_summary.csv",index=False)
    ck.mark_done("04_analysis",artifacts=[str(result_dir/"bootstrap_effect.json"),str(av)]); _record_runtime(paths,args.dataset,"analysis",t0)
    print(json.dumps(effect,indent=2))

def cmd_grid(args):
    t0=time.perf_counter()
    paths=get_paths(args.drive_root,args.local_root); ck=CheckpointStore(paths.checkpoints,args.dataset); _stage(args); m,folds=_load_manifest(paths,args.dataset); X,G0,_=_load_features(paths,args.dataset,need_lbp=False)
    result_dir=paths.results/args.dataset/"paper_v1_0"; rows=[]; K=m.class_id.nunique(); base_pred=pd.read_csv(result_dir/"outer_predictions.csv"); P0=_prob(base_pred,"CoSS_DF",K)
    for oy,ox in [(0,0),(0,4),(4,0),(4,4)]:
        if (oy,ox)==(0,0): G=G0; P=P0
        else:
            fp=extract_coordinate_energy_chunked(m,ck.artifact_dir("features","spectral"),chunk_size=args.spectral_chunk,m=8,offset=(oy,ox)); G=np.load(fp)["G"].astype(float); P=variant_spectral_outer(args.dataset,m,folds,X,G,result_dir/"selected_configs.csv")
        mt=all_metrics(m.class_id.values,P=P)
        drift=float(np.mean(np.linalg.norm(G.reshape(len(G),-1)-G0.reshape(len(G0),-1),axis=1)/(np.linalg.norm(G0.reshape(len(G0),-1),axis=1)+1e-300)))
        rows.append({"offset_y":oy,"offset_x":ox,"descriptor_drift":drift,**mt})
    pd.DataFrame(rows).to_csv(result_dir/"paper_tables"/"grid_offset_sensitivity.csv",index=False); ck.mark_done("05_grid_sensitivity"); _record_runtime(paths,args.dataset,"grid_sensitivity",t0)

def cmd_sensitivity(args):
    t0=time.perf_counter()
    if args.dataset != "kth_tips2b": raise RuntimeError("eta/gamma sensitivity is prespecified for KTH-TIPS2b only")
    paths=get_paths(args.drive_root,args.local_root); ck=CheckpointStore(paths.checkpoints,args.dataset); _stage(args); m,folds=_load_manifest(paths,args.dataset); X,G,_=_load_features(paths,args.dataset,need_lbp=False)
    result_dir=paths.results/args.dataset/"paper_v1_0"; selected=result_dir/"selected_configs.csv"; rows=[]
    for kind in ["eta","gamma"]:
        for value in [1e-8,1e-6,1e-4,1e-2]:
            eta=value if kind=="eta" else 1e-6; gamma=value if kind=="gamma" else 1e-6
            P=variant_spectral_outer(args.dataset,m,folds,X,G,selected,eta=eta,gamma=gamma)
            rows.append({"parameter":kind,"value":value,**all_metrics(m.class_id.values,P=P)})
    out=result_dir/"paper_tables"/"eta_gamma_sensitivity.csv"; pd.DataFrame(rows).to_csv(out,index=False); ck.mark_done("06_eta_gamma_sensitivity",artifacts=[str(out)]); _record_runtime(paths,args.dataset,"eta_gamma_sensitivity",t0)

def _load_manifest_nostage(paths,dataset):
    out=paths.checkpoints/dataset/"audit"
    return pd.read_csv(out/"manifest.csv"),pd.read_csv(out/"fold_assignments.csv")

def cmd_geometry_control(args):
    if args.dataset not in {"kth_tips2b","fmd","kylberg"}:
        raise RuntimeError("geometry_control is only for the three primary datasets")
    paths=get_paths(args.drive_root,args.local_root)
    m,folds=_load_manifest_nostage(paths,args.dataset)
    X,G,_=_load_features(paths,args.dataset,need_lbp=False)
    rd=paths.results/args.dataset/"paper_v1_0"
    out=rd/"geometry_control"
    metrics,effect,sel=run_geometry_control(args.dataset,m,folds,X,G,rd,out)
    gd=np.load(paths.checkpoints/args.dataset/"features"/"spectral"/"coordinate_energy_o0_0.npz")
    S=np.asarray(gd["S_E"],float)
    atomic_json(out/"spectral_energy_boundary.json",{
        "dataset":args.dataset,"min_S_E":float(S.min()),"zero_count":int(np.sum(S<=0)),"n":int(len(S))
    })
    print(metrics.to_string(index=False))
    print(json.dumps(effect,indent=2))

def cmd_geometry_control_collect(args):
    paths=get_paths(args.drive_root,args.local_root)
    out=paths.results/"geometry_control_global"; out.mkdir(parents=True,exist_ok=True)
    mets=[]; eff=[]; sels=[]; energy=[]
    for ds in ["kth_tips2b","fmd","kylberg"]:
        rd=paths.results/ds/"paper_v1_0"/"geometry_control"
        if not (rd/"geometry_control_metrics.csv").exists():
            raise RuntimeError(f"Missing geometry control for {ds}; run cossdf geometry_control --dataset {ds}")
        mets.append(pd.read_csv(rd/"geometry_control_metrics.csv"))
        sels.append(pd.read_csv(rd/"geometry_control_selected.csv").assign(dataset=ds))
        eff.append(json.loads((rd/"B5_minus_B4W_bootstrap.json").read_text()))
        energy.append(json.loads((rd/"spectral_energy_boundary.json").read_text()))
    pd.concat(mets,ignore_index=True).to_csv(out/"geometry_control_metrics_all.csv",index=False)
    pd.concat(sels,ignore_index=True).to_csv(out/"geometry_control_selected_all.csv",index=False)
    pd.DataFrame(eff).to_csv(out/"B5_minus_B4W_effects.csv",index=False)
    pd.DataFrame(energy).to_csv(out/"spectral_energy_boundary.csv",index=False)
    atomic_json(out/"geometry_control_summary.json",{"effects":eff,"energy":energy})
    print(out)


def cmd_directional_fusion_controls(args):
    if args.dataset not in {"kth_tips2b","fmd","kylberg"}:
        raise RuntimeError("directional_fusion_controls is only for the three primary datasets")
    paths=get_paths(args.drive_root,args.local_root)
    m,folds=_load_manifest_nostage(paths,args.dataset)
    X,G,_=_load_features(paths,args.dataset,need_lbp=False)
    rd=paths.results/args.dataset/"paper_v1_0"
    out=rd/"directional_fusion_controls"
    met,eff,diag,sel=run_directional_fusion_controls(args.dataset,m,folds,X,G,rd,out)
    print(met.to_string(index=False))
    print("\\nDirectional/fusion control effects:\\n",eff.to_string(index=False))
    print("\\nSynergy diagnostics:\\n",diag.to_string(index=False))


def cmd_directional_fusion_controls_collect(args):
    paths=get_paths(args.drive_root,args.local_root)
    out=paths.results/"directional_fusion_controls_global"; out.mkdir(parents=True,exist_ok=True)
    mets=[]; effects=[]; diags=[]; sels=[]
    for ds in ["kth_tips2b","fmd","kylberg"]:
        rd=paths.results/ds/"paper_v1_0"/"directional_fusion_controls"
        if not (rd/"directional_fusion_controls_metrics.csv").exists():
            raise RuntimeError(f"Missing directional/fusion control for {ds}; run cossdf directional_fusion_controls --dataset {ds}")
        mets.append(pd.read_csv(rd/"directional_fusion_controls_metrics.csv"))
        effects.append(pd.read_csv(rd/"directional_fusion_controls_effects.csv"))
        diags.append(pd.read_csv(rd/"directional_fusion_controls_synergy_diagnostics.csv"))
        sels.append(pd.read_csv(rd/"directional_fusion_controls_selected.csv").assign(dataset=ds))
    pd.concat(mets,ignore_index=True).to_csv(out/"directional_fusion_controls_metrics_all.csv",index=False)
    pd.concat(effects,ignore_index=True).to_csv(out/"directional_fusion_controls_effects_all.csv",index=False)
    pd.concat(diags,ignore_index=True).to_csv(out/"directional_fusion_controls_synergy_diagnostics_all.csv",index=False)
    pd.concat(sels,ignore_index=True).to_csv(out/"directional_fusion_controls_selected_all.csv",index=False)
    atomic_json(out/"directional_fusion_controls_summary.json",{
        "purpose":"Secondary geometry/anisotropy/strong-expert fusion controls",
        "datasets":["kth_tips2b","fmd","kylberg"],"directional_sectors":2,
        "comparisons":["DIR_MATCHED-B5","DIR_TUNED-B5","C3-fusion-C3S","LR-ILR-LR-linear"]})
    print(out)



def _resource_snapshot():
    import psutil
    rec={'rss_gb':psutil.Process().memory_info().rss/1024**3}
    try:
        import torch
        rec['gpu_peak_gb']=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0.0
    except Exception:
        rec['gpu_peak_gb']=None
    return rec


def cmd_additional_validation(args):
    """Additional representation/deep-baseline validation.

    This command deliberately stages the dataset because DWT/Gabor and the two
    modern frozen backbones require the image pixels. Original primary folds and
    all previous predictions remain unchanged.
    """
    if args.dataset not in {'kth_tips2b','fmd','kylberg'}:
        raise RuntimeError('additional_validation is only for the three primary datasets')
    paths=get_paths(args.drive_root,args.local_root); ck=CheckpointStore(paths.checkpoints,args.dataset)
    _stage(args); m,folds=_load_manifest(paths,args.dataset); Xres,G,_=_load_features(paths,args.dataset,need_lbp=False)
    out=paths.results/args.dataset/'paper_v1_0'/'additional_validation'; out.mkdir(parents=True,exist_ok=True)
    froot=ck.artifact_dir('features','additional_validation')
    timing_path=out/'additional_validation_timing_memory.json'; timing=json.loads(timing_path.read_text()) if timing_path.exists() else {}

    # Fixed handcrafted spectral representations.
    hf=froot/'handcrafted'/'handcrafted_energy.npz'
    if not hf.exists():
        t0=time.perf_counter(); extract_handcrafted_energy_chunked(m,froot/'handcrafted',chunk_size=args.spectral_chunk,m=8)
        timing['handcrafted_features']={'seconds':time.perf_counter()-t0,**_resource_snapshot()}; atomic_json(timing_path,timing)
    hd=np.load(hf); E_dwt=hd['E_DWT'].astype(float); E_gabor=hd['E_GABOR'].astype(float)

    # Frozen modern embeddings. Each is separately checkpointed and resumable.
    for backbone in ['efficientnet_b0','vit_b_16']:
        bp=froot/backbone/'embeddings.npz'
        if not bp.exists():
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            except Exception: pass
            t0=time.perf_counter(); extract_backbone_chunked(m,froot/backbone,backbone,batch_size=args.modern_batch_size,chunk_size=args.feature_chunk)
            timing[f'{backbone}_features']={'seconds':time.perf_counter()-t0,**_resource_snapshot()}; atomic_json(timing_path,timing)
    Xeff=np.load(froot/'efficientnet_b0'/'embeddings.npz')['X'].astype(float)
    Xvit=np.load(froot/'vit_b_16'/'embeddings.npz')['X'].astype(float)

    control_predictions_path=paths.results/args.dataset/'paper_v1_0'/'directional_fusion_controls'/'directional_fusion_controls_outer_predictions.csv'
    if not control_predictions_path.exists(): raise RuntimeError('additional_validation requires the completed directional_fusion_controls predictions for the ResNet18 C3S comparator')
    control_predictions=pd.read_csv(control_predictions_path)
    t0=time.perf_counter(); met,eff,sel=run_additional_validation(args.dataset,m,folds,G,E_dwt,E_gabor,Xeff,Xvit,control_predictions,out,bootstrap_B=10000)
    timing['statistical_modeling']={'seconds':time.perf_counter()-t0,**_resource_snapshot()}; atomic_json(timing_path,timing)

    # CLR/ILR identity verification on a fixed R=6 DCT composition.
    from .features.spectral import spectral_from_G
    _,P6,_=spectral_from_G(G,6,1e-6)
    sanity={'dataset':args.dataset,**clr_ilr_sanity(P6,max_pairs=2000,seed=2026)}; atomic_json(out/'clr_ilr_sanity.json',sanity)
    print(met.to_string(index=False)); print('\nSecondary paired effects:\n',eff.to_string(index=False)); print('\nCLR/ILR sanity:',json.dumps(sanity,indent=2))


def cmd_additional_validation_collect(args):
    paths=get_paths(args.drive_root,args.local_root); out=paths.results/'additional_validation_global'; out.mkdir(parents=True,exist_ok=True)
    mets=[]; effects=[]; sels=[]; timing=[]; sanity=[]
    for ds in ['kth_tips2b','fmd','kylberg']:
        rd=paths.results/ds/'paper_v1_0'/'additional_validation'
        if not (rd/'additional_validation_metrics.csv').exists(): raise RuntimeError(f'Missing additional validation for {ds}; run cossdf additional_validation --dataset {ds}')
        mets.append(pd.read_csv(rd/'additional_validation_metrics.csv'))
        effects.append(pd.read_csv(rd/'additional_validation_effects.csv'))
        sels.append(pd.read_csv(rd/'additional_validation_selected.csv').assign(dataset=ds))
        tr=json.loads((rd/'additional_validation_timing_memory.json').read_text());
        for stage,rec in tr.items(): timing.append({'dataset':ds,'stage':stage,**rec})
        sanity.append(json.loads((rd/'clr_ilr_sanity.json').read_text()))
    pd.concat(mets,ignore_index=True).to_csv(out/'additional_validation_metrics_all.csv',index=False)
    pd.concat(effects,ignore_index=True).to_csv(out/'additional_validation_effects_all.csv',index=False)
    pd.concat(sels,ignore_index=True).to_csv(out/'additional_validation_selected_all.csv',index=False)
    pd.DataFrame(timing).to_csv(out/'additional_validation_timing_memory_all.csv',index=False)
    pd.DataFrame(sanity).to_csv(out/'clr_ilr_sanity_all.csv',index=False)
    atomic_json(out/'additional_validation_summary.json',{
        'purpose':'Additional theory/representation/modern-baseline validation',
        'datasets':['kth_tips2b','fmd','kylberg'],
        'status':'secondary descriptive evidence; nominal paired 95% intervals without multiplicity adjustment',
        'representations':['DCT-ILR-LR','fixed 2-level Haar DWT-ILR-LR','fixed 8-atom local Gabor-ILR-LR'],
        'modern_baselines':['EfficientNet-B0 frozen + logistic','ViT-B/16 frozen + logistic'],
        'clr':'numerical identity check only, not a predictive competitor'
    })
    print(out)

def cmd_collect(args):
    paths=get_paths(args.drive_root,args.local_root); global_out=paths.results/"paper_global"; (global_out/"figures").mkdir(parents=True,exist_ok=True); (global_out/"paper_tables").mkdir(parents=True,exist_ok=True)
    metrics=[]; effects=[]
    for ds in ["kth_tips2b","fmd","kylberg"]:
        rd=paths.results/ds/"paper_v1_0"
        if (rd/"metrics.csv").exists(): metrics.append(pd.read_csv(rd/"metrics.csv"))
        if (rd/"bootstrap_effect.json").exists():
            e=json.loads((rd/"bootstrap_effect.json").read_text()); ci=e["delta_macro_f1_ci"]; effects.append({"dataset":ds,"delta_macro_f1":e["delta_macro_f1"],"delta_nll":e["delta_nll"],"f1_ci_low":ci[0],"f1_ci_high":ci[1],"nll_ci_low":e["delta_nll_ci"][0],"nll_ci_high":e["delta_nll_ci"][1]})
    if metrics: pd.concat(metrics,ignore_index=True).to_csv(global_out/"paper_tables"/"all_dataset_metrics.csv",index=False)
    if effects:
        pd.DataFrame(effects).to_csv(global_out/"paper_tables"/"primary_effect_sizes.csv",index=False); effect_size_plot([e for e in effects if e["dataset"] in ["kth_tips2b","fmd","kylberg"]],global_out/"figures"/"effect_sizes_macro_f1.pdf")
    atomic_json(global_out/"paper_results.json",{"effects":effects})
    print(global_out)

def cmd_all(args):
    cmd_prepare(args); cmd_audit(args); cmd_features(args); cmd_run(args); cmd_analyze(args)
    if args.run_grid: cmd_grid(args)
    if args.run_sensitivity and args.dataset=="kth_tips2b": cmd_sensitivity(args)

def build_parser():
    p=argparse.ArgumentParser(prog="cossdf",description="CoSS-DF Colab/Drive resumable pipeline")
    p.add_argument("command",choices=["prepare","audit","features","run","analyze","grid","sensitivity","geometry_control","geometry_control_collect","directional_fusion_controls","directional_fusion_controls_collect","additional_validation","additional_validation_collect","collect","all"]); p.add_argument("--dataset",required=False,choices=["kth_tips2b","fmd","kylberg"])
    p.add_argument("--drive-root",default=None); p.add_argument("--local-root",default=None); p.add_argument("--source-url",default=None)
    p.add_argument("--force-stage",action="store_true"); p.add_argument("--allow-audit-mismatch",action="store_true")
    p.add_argument("--batch-size",type=int,default=64); p.add_argument("--modern-batch-size",type=int,default=32); p.add_argument("--feature-chunk",type=int,default=512); p.add_argument("--spectral-chunk",type=int,default=256); p.add_argument("--skip-lbp",action="store_true"); p.add_argument("--run-grid",action="store_true"); p.add_argument("--run-sensitivity",action="store_true")
    return p

def main():
    args=build_parser().parse_args(); seed_everything(2026)
    if args.command not in {"collect","geometry_control_collect","directional_fusion_controls_collect","additional_validation_collect"} and not args.dataset: raise SystemExit("--dataset is required for this command")
    {"prepare":cmd_prepare,"audit":cmd_audit,"features":cmd_features,"run":cmd_run,"analyze":cmd_analyze,"grid":cmd_grid,"sensitivity":cmd_sensitivity,"geometry_control":cmd_geometry_control,"geometry_control_collect":cmd_geometry_control_collect,"directional_fusion_controls":cmd_directional_fusion_controls,"directional_fusion_controls_collect":cmd_directional_fusion_controls_collect,"additional_validation":cmd_additional_validation,"additional_validation_collect":cmd_additional_validation_collect,"collect":cmd_collect,"all":cmd_all}[args.command](args)

if __name__=="__main__": main()
