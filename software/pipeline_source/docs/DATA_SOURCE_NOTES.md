# Dataset source notes

The reported study uses three public benchmark datasets only: KTH-TIPS2b, FMD, and Kylberg Texture Dataset v1.0. Raw images are not redistributed in this repository.

- **KTH-TIPS2b**: obtained from the official KTH/CVAP distribution page.
- **FMD**: obtained from the official MIT distribution page.
- **Kylberg Texture Dataset v1.0**: obtained from the official Kylberg distribution. The reported protocol uses the full non-rotated release with 28 classes, 160 images per class, and source-surface identifiers a/b/c/d.

The downloader attempts the official sources defined in `configs/datasets.yaml`. If the Kylberg hosting layout changes, provide the official non-rotated archive explicitly with `--source-url`, or place the archive at the path reported by the command and rerun.

Dataset integrity is checked before modeling. The retained `integrity/` artifacts at the repository root contain the manifests, fold assignments, exact-duplicate checks, and near-duplicate candidate reports used for the reported study.
