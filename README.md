# coss-df

Reproducibility code and retained audit artifacts for the manuscript

**Compositional DCT Spectral Geometry for Material Image Classification: Representation Analysis and Probabilistic Evidence Fusion**

by **Thien Nguyen-Chi and Tai Vo-Van**.

The repository is organized for direct inspection of the reported computational pipeline. It contains ordinary Python source files, fixed experiment configuration, tests, dataset-integrity artifacts, compact reference results, and the recorded runtime environment. Raw benchmark images are not redistributed.

## Repository contents

```text
coss-df/
├── software/pipeline_source/          # browsable Python implementation
├── integrity/                         # manifests, folds, duplicate audits
├── reference_results/                 # compact machine-readable reported summaries
├── environment/
│   └── reported_environment.json      # runtime used for the primary runs
├── docs/
│   ├── SOURCE_CODE_INDEX.md
│   ├── TEST_REPORT.txt
│   └── ARCHIVAL_SOFTWARE_SHA256.txt
├── CITATION.cff
├── .gitignore
└── README.md
```

## Reported datasets

The study uses exactly three primary public datasets:

- **KTH-TIPS2b** — 4752 images, 11 classes; grouped by physical specimen.
- **FMD** — 1000 images, 10 classes; stratified image-level folds.
- **Kylberg Texture Dataset v1.0** — 4480 images, 28 classes; grouped by original source surface.

KTH-TIPS2b and Kylberg are evaluated with provenance-aware grouping so images derived from the same physical source do not cross outer folds.

## Recommended code-reading path

```text
software/pipeline_source/cossdf/features/spectral.py
software/pipeline_source/cossdf/models/transforms.py
software/pipeline_source/cossdf/models/density.py
software/pipeline_source/cossdf/models/fusion.py
software/pipeline_source/cossdf/geometry_control.py
software/pipeline_source/cossdf/directional_fusion_controls.py
software/pipeline_source/cossdf/features/alternative_spectral.py
software/pipeline_source/cossdf/additional_validation.py
software/pipeline_source/cossdf/analysis/bootstrap.py
software/pipeline_source/cossdf/runner.py
```

## Installation and validation

```bash
cd software/pipeline_source
python -m pip install -r requirements-colab.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m compileall -q cossdf
pytest -q
```

Expected test result:

```text
25 passed
```

The package exposes the `cossdf` command-line entry point.

## Main execution

```bash
export COSSDF_DRIVE_ROOT=/content/drive/MyDrive/CoSS_DF
export COSSDF_LOCAL_ROOT=/content/cossdf_work

cossdf all --dataset kth_tips2b
cossdf all --dataset fmd
cossdf all --dataset kylberg
cossdf collect
```

Additional reported controls are executed with `geometry_control`, `directional_fusion_controls`, and `additional_validation`, followed by their corresponding `*_collect` commands.

## Data acquisition and integrity

Dataset source definitions are in `software/pipeline_source/configs/datasets.yaml`. The public source uses the official dataset distributions; if a host layout changes, an official archive may be supplied explicitly with `--source-url`.

The `integrity/` directory retains the exact manifests, fold assignments, SHA-256 duplicate reports, and near-duplicate candidate reports used for the reported study.

## Reference results

`reference_results/` contains compact CSV/JSON summaries for the primary comparison, matched geometry control, directional/fusion analyses, alternative spectral representations, modern frozen backbones, CLR/ILR numerical checks, and runtime/memory summaries. These files are audit targets, not tuning targets.

## Reported environment

The primary runs used the environment recorded in `environment/reported_environment.json`, including Python, NumPy, SciPy, scikit-learn, PyTorch, torchvision, and Tesla T4 GPU versions.

## Supplementary Software S1

The journal submission also contains an archival Supplementary Software S1 package with retained outer-test probabilities and additional machine-readable artifacts. Its SHA-256 digest is recorded in `docs/ARCHIVAL_SOFTWARE_SHA256.txt`. GitHub is intended for direct browsing and execution; Software S1 preserves the submission-time archival snapshot.

## Citation

Citation metadata are provided in `CITATION.cff`. The journal DOI/citation can be added after publication.

## License and use

No open-source license is assigned at the peer-review stage. The repository is made public for inspection, verification, and reproducibility of the accompanying research manuscript. Unless otherwise stated, normal copyright restrictions apply.

## Contact

Technical reproducibility issues may be reported through the GitHub issue tracker.
