from __future__ import annotations
import os, shutil, subprocess
from pathlib import Path

def stage_dataset_to_ssd(src: str|Path, dst: str|Path, force=False):
    src=Path(src); dst=Path(dst)
    if not src.exists(): raise FileNotFoundError(src)
    if dst.exists() and not force and (dst/".stage_complete").exists():
        print(f"[stage] SSD copy already present: {dst}"); return dst
    shutil.rmtree(dst,ignore_errors=True); dst.mkdir(parents=True,exist_ok=True)
    # rsync is fast and gives a robust directory copy in Colab.
    if shutil.which("rsync"):
        rc=subprocess.call(["rsync","-a","--delete",str(src)+"/",str(dst)+"/"])
        if rc!=0: raise RuntimeError("rsync staging failed")
    else:
        shutil.rmtree(dst,ignore_errors=True); shutil.copytree(src,dst)
    (dst/".stage_complete").write_text("ok\n")
    return dst
