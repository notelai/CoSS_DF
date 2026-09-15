from __future__ import annotations
from pathlib import Path
import json, pandas as pd
from ..utils import atomic_json

def export_summary(dataset, pred_df, metrics_df, selected_df, out_dir):
    out=Path(out_dir); (out/"paper_tables").mkdir(parents=True,exist_ok=True)
    pred_df.to_csv(out/"outer_predictions.csv",index=False)
    metrics_df.to_csv(out/"paper_tables"/f"{dataset}_metrics.csv",index=False)
    selected_df.to_csv(out/"paper_tables"/f"{dataset}_selected_configs.csv",index=False)
    obj={"dataset":dataset,"metrics":metrics_df.to_dict("records"),"selected_configs":selected_df.to_dict("records")}
    atomic_json(out/"paper_results.json",obj)
