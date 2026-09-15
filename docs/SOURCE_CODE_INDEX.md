# Source code index

This page maps the scientific components of the manuscript to the expanded Python implementation in

```text
software/pipeline_source/
```

## Core representation and modeling

- `cossdf/features/spectral.py` — centered block DCT, non-DC coordinate energies, radial-band aggregation, simplex closure and spectral feature construction.
- `cossdf/models/transforms.py` — PCA/ILR transformations and regularized whitening.
- `cossdf/models/density.py` — class-conditional KDE and Gaussian density models.
- `cossdf/models/fusion.py` — posterior linear pooling and logarithmic opinion pooling.

## Controlled representation analyses

- `cossdf/geometry_control.py` — matched linear-simplex versus ILR geometry control.
- `cossdf/directional_fusion_controls.py` — two-sector directional analysis, classifier-independent geometry controls, strong spatial/spectral experts and evidence-fusion diagnostics.
- `cossdf/features/alternative_spectral.py` — fixed Haar-DWT/Gabor energy features and CLR/ILR utilities.
- `cossdf/additional_validation.py` — alternative-representation comparisons, frozen EfficientNet-B0 and ViT-B/16 baselines, CLR/ILR numerical checks, runtime/memory logging.

## Evaluation and uncertainty

- `cossdf/runner.py` — nested inner selection and outer-fold evaluation for the primary pipeline.
- `cossdf/analysis/bootstrap.py` — paired bootstrap uncertainty analysis.
- `cossdf/metrics.py` — classification and probabilistic metrics.
- `cossdf/analysis/paper.py` — paper-oriented summary tables and retained outputs.

## Data integrity and splits

- `cossdf/datasets/download.py` — public-source acquisition.
- `cossdf/datasets/adapters.py` — dataset-specific parsing and provenance extraction.
- `cossdf/datasets/audit.py` — image validation, SHA-256 duplicate checks and integrity summaries.
- `cossdf/datasets/splits.py` — grouped/stratified outer and inner split construction.

## Configuration and execution

- `configs/frozen_config.json` — fixed primary modeling configuration.
- `configs/datasets.yaml` — public dataset registry and expected release counts.
- `cossdf/cli.py` — command-line interface.
- `scripts/` — convenience execution scripts.
- `colab/CoSS_DF_Run_Pipeline.ipynb` — Colab-oriented driver.
- `tests/` — unit and smoke tests.

## Validation

From `software/pipeline_source/`:

```bash
python -m pip install -r requirements-colab.txt
python -m pip install -e .
python -m compileall -q cossdf
pytest -q
```

Expected result:

```text
25 passed
```
