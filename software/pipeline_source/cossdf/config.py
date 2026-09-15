from __future__ import annotations
import json, os
from dataclasses import dataclass
from pathlib import Path
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Paths:
    drive_root: Path
    local_root: Path
    @property
    def datasets_drive(self): return self.drive_root / "datasets"
    @property
    def datasets_local(self): return self.local_root / "datasets"
    @property
    def checkpoints(self): return self.drive_root / "checkpoints"
    @property
    def results(self): return self.drive_root / "results"
    @property
    def logs(self): return self.drive_root / "logs"


def default_drive_root() -> Path:
    return Path(os.environ.get("COSSDF_DRIVE_ROOT", "/content/drive/MyDrive/CoSS_DF"))

def default_local_root() -> Path:
    return Path(os.environ.get("COSSDF_LOCAL_ROOT", "/content/cossdf_work"))

def get_paths(drive_root=None, local_root=None) -> Paths:
    p=Paths(Path(drive_root or default_drive_root()), Path(local_root or default_local_root()))
    for x in [p.drive_root,p.local_root,p.datasets_drive,p.datasets_local,p.checkpoints,p.results,p.logs]: x.mkdir(parents=True,exist_ok=True)
    return p

def load_frozen(path=None):
    p=Path(path) if path else PACKAGE_ROOT/"configs"/"frozen_config.json"
    return json.loads(p.read_text())

def load_datasets(path=None):
    p=Path(path) if path else PACKAGE_ROOT/"configs"/"datasets.yaml"
    return yaml.safe_load(p.read_text())
