from __future__ import annotations
from pathlib import Path
from .utils import atomic_json, read_json, now_iso, stable_hash

class CheckpointStore:
    """Drive-backed atomic checkpoint metadata. Artifacts themselves also live on Drive."""
    def __init__(self, root: str|Path, dataset: str):
        self.root=Path(root)/dataset; self.root.mkdir(parents=True,exist_ok=True)
    def stage_file(self, stage): return self.root/f"{stage}.done.json"
    def is_done(self, stage, signature=None):
        d=read_json(self.stage_file(stage))
        return bool(d and d.get("status")=="done" and (signature is None or d.get("signature")==signature))
    def mark_done(self, stage, signature=None, artifacts=None, extra=None):
        atomic_json(self.stage_file(stage), {"status":"done","stage":stage,"timestamp":now_iso(),"signature":signature,"artifacts":artifacts or [],"extra":extra or {}})
    def invalidate(self, stage):
        p=self.stage_file(stage)
        if p.exists(): p.unlink()
    def artifact_dir(self, *parts):
        p=self.root.joinpath(*parts); p.mkdir(parents=True,exist_ok=True); return p
    def signature(self, **kwargs): return stable_hash(kwargs)
