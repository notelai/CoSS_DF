from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def reliability_plot(items,y,out):
    fig,ax=plt.subplots(figsize=(6,5)); ax.plot([0,1],[0,1],"--",linewidth=1)
    for name,P in items.items():
        conf=P.max(1); pred=P.argmax(1); order=np.argsort(conf); bins=np.array_split(order,15)
        xs=[]; ys=[]
        for b in bins:
            if len(b): xs.append(conf[b].mean()); ys.append((pred[b]==y[b]).mean())
        ax.plot(xs,ys,marker="o",label=name)
    ax.set(xlabel="Confidence",ylabel="Accuracy",xlim=(0,1),ylim=(0,1)); ax.legend(); fig.tight_layout();
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,bbox_inches="tight"); plt.close(fig)

def effect_size_plot(rows,out):
    labels=[r["dataset"] for r in rows]; y=np.arange(len(rows)); vals=[r["delta_macro_f1"] for r in rows]; lo=[r["f1_ci_low"] for r in rows]; hi=[r["f1_ci_high"] for r in rows]
    fig,ax=plt.subplots(figsize=(6,4)); ax.axvline(0,linewidth=1); ax.errorbar(vals,y,xerr=[np.array(vals)-np.array(lo),np.array(hi)-np.array(vals)],fmt="o")
    ax.set_yticks(y,labels); ax.set_xlabel("Δ Macro-F1: CoSS-DF − B1"); fig.tight_layout(); out=Path(out); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,bbox_inches="tight"); plt.close(fig)
