"""Recreate the four-dataset coverage comparison from the stored summary CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
s=pd.read_csv(root/"results"/"known_prior_real_summary.csv")
datasets=["digits","diabetes","wine","breast_cancer"]
labels={"digits":"Digits","diabetes":"Diabetes","wine":"Wine","breast_cancer":"Breast Cancer"}
methods=[
    ("Pooled","Pooled"),
    ("Uncorrected proxy inversion","Proxy plug-in"),
    ("Certified proxy inversion","Proxy certificate"),
    ("Latent-label oracle (DKW)","Latent oracle + DKW"),
]
x=np.arange(len(datasets)); width=.18
fig,ax=plt.subplots(figsize=(9,4.6))
for i,(method,label) in enumerate(methods):
    v=[float(s[(s.dataset==d)&(s.method==method)]["coverage"].iloc[0]) for d in datasets]
    ax.bar(x+(i-1.5)*width,v,width,label=label)
ax.axhline(.90,linestyle="--",linewidth=1.2)
ax.set_xticks(x);ax.set_xticklabels([labels[d] for d in datasets])
ax.set_ylabel("Mean held-out coverage")
ax.set_ylim(.83,1.0)
ax.legend(fontsize=9,loc="upper left",ncol=2)
fig.tight_layout()
out=root/"figures";out.mkdir(exist_ok=True)
fig.savefig(out/"known_prior_comparison.pdf",bbox_inches="tight")
fig.savefig(out/"known_prior_comparison.png",dpi=220,bbox_inches="tight")
plt.close(fig)
