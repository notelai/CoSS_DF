# Resume after a Colab disconnect

1. Reconnect to a GPU runtime and mount Google Drive.
2. Clone the repository again into `/content/coss-df`, or reuse a fresh clone.
3. From the repository root, run `python software/pipeline_source/scripts/bootstrap_colab.py`.
4. Rerun the same command as before, for example `cossdf all --dataset kth_tips2b`.
5. The active dataset is restaged from Drive to the fresh `/content` SSD.
6. Existing Drive checkpoints are detected and reused; completed chunks and folds are not recomputed.

Do not delete `/content/drive/MyDrive/CoSS_DF/checkpoints` unless a complete rerun is intended.
