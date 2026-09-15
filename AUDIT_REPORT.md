# GitHub release audit

This repository snapshot was checked before public upload.

- Python source compiles successfully.
- Unit/smoke tests pass: **25 passed**.
- The package builds as `cossdf` version **1.0.6** and includes all nested source packages.
- The public interface contains only the three datasets reported in the manuscript: KTH-TIPS2b, FMD, and Kylberg Texture Dataset v1.0.
- Public dataset acquisition is restricted to official sources or an explicitly supplied official archive URL.
- Dataset-integrity manifests/folds and compact reported result summaries are included.
- The actual primary-run environment is recorded in `environment/reported_environment.json`.
- No API keys, passwords, private keys, personal home-directory paths, or stored notebook outputs are included.
- Internal conference/revision history and obsolete development documents are not included.
- The journal Supplementary Software S1 SHA-256 digest matches the archived submission package.

One intentional placeholder remains: `GITHUB_URL` in the Colab notebook. Replace it after the GitHub repository has been created; see `RELEASE_CHECKLIST.md`.

The GitHub source tree contains paper-facing documentation/naming cleanup and removal of unused development-only dataset/download paths. The statistical methods, fixed configuration, reported dataset protocols, and analysis implementations are unchanged. The exact submission-time archival package remains preserved as Supplementary Software S1.
