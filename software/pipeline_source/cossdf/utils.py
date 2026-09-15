from __future__ import annotations
import hashlib, json, os, random, tempfile, time
from pathlib import Path
from typing import Any
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def seed_everything(seed: int = 2026) -> None:
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def atomic_json(path: str | Path, obj: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def read_json(path: str | Path, default=None):
    p=Path(path)
    if not p.exists(): return default
    return json.loads(p.read_text(encoding="utf-8"))

def sha256_file(path: str | Path, chunk: int = 1024*1024) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def stable_hash(obj: Any) -> str:
    b=json.dumps(obj, sort_keys=True, default=str).encode(); return hashlib.sha256(b).hexdigest()[:16]

def list_images(root: str | Path):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def now_iso():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()

def timer(): return time.perf_counter()
