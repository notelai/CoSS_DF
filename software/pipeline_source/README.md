# CoSS-DF computational pipeline

Expanded Python source for the computational study accompanying

**Compositional DCT Spectral Geometry for Material Image Classification: Representation Analysis and Probabilistic Evidence Fusion**.

The reported study uses KTH-TIPS2b, FMD, and Kylberg Texture Dataset v1.0. Raw images are not redistributed. Dataset sources and expected release counts are defined in `configs/datasets.yaml`.

## Source layout

```text
pipeline_source/
├── cossdf/
│   ├── datasets/
│   ├── features/
│   ├── models/
│   ├── analysis/
│   ├── geometry_control.py
│   ├── directional_fusion_controls.py
│   ├── additional_validation.py
│   ├── runner.py
│   └── cli.py
├── configs/
├── colab/
├── docs/
├── scripts/
├── tests/
├── pyproject.toml
├── requirements-colab.txt
└── README.md
```

## Scientific source map

| Scientific component | Main source file |
|---|---|
| Centered DCT and spectral-energy composition | `cossdf/features/spectral.py` |
| ILR/PCA and regularized whitening | `cossdf/models/transforms.py` |
| Class-conditional KDE / Gaussian density models | `cossdf/models/density.py` |
| Linear pooling and logarithmic opinion pooling | `cossdf/models/fusion.py` |
| Matched linear-simplex vs ILR geometry control | `cossdf/geometry_control.py` |
| Directional, logistic-geometry, and fusion controls | `cossdf/directional_fusion_controls.py` |
| Haar-DWT/Gabor and CLR/ILR utilities | `cossdf/features/alternative_spectral.py` |
| EfficientNet/ViT and additional validation | `cossdf/additional_validation.py` |
| Paired bootstrap | `cossdf/analysis/bootstrap.py` |
| Nested outer/inner evaluation | `cossdf/runner.py` |
| Integrity / duplicate checks | `cossdf/datasets/audit.py` |
| Provenance-aware split construction | `cossdf/datasets/splits.py` |

## Installation

Google Colab already provides PyTorch and torchvision. From this directory:

```bash
python -m pip install -r requirements-colab.txt
python -m pip install --no-build-isolation --no-deps -e .
```

## Validation

```bash
python -m compileall -q cossdf
pytest -q
```

Expected result for this public source tree:

```text
25 passed
```

## Primary datasets

```text
kth_tips2b
fmd
kylberg
```

Example:

```bash
export COSSDF_DRIVE_ROOT=/content/drive/MyDrive/CoSS_DF
export COSSDF_LOCAL_ROOT=/content/cossdf_work
cossdf all --dataset kth_tips2b
```

Repeat for `fmd` and `kylberg`, then run:

```bash
cossdf collect
```

## Controlled analyses

```bash
cossdf geometry_control --dataset kth_tips2b
cossdf directional_fusion_controls --dataset kth_tips2b
cossdf additional_validation --dataset kth_tips2b
```

Repeat each analysis for all three reported datasets and run the corresponding `*_collect` command.

## Colab and resume

`colab/CoSS_DF_Run_Pipeline.ipynb` provides a Colab-oriented driver. Replace `GITHUB_URL` in its setup cell after the repository has been created. Persistent artifacts are written to Google Drive and the active dataset is staged to `/content` for computation. See `docs/RESUME_GUIDE.md`.

## Archival software snapshot

The journal submission includes Supplementary Software S1, which preserves the submission-time archival software/results package. The GitHub source tree is the browsable paper-aligned implementation; retained summary results and integrity artifacts are exposed at the repository root for convenient audit.
