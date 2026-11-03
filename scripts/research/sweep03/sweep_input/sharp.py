import pickle, numpy as np, pandas as pd
from pathlib import Path
d=pickle.load(open(Path(__file__).parent/"tables.pkl","rb")); ent,img,P,G=d["ent"],d["img"],d["P"],d["G"]
ent["grp"]=ent.key.str[:2]; ent=ent.merge(img[["stem","lap_var","hp_std","flat_std"]],on="stem")
pq=lambda x: x.siou.sum()/(x.tp.sum()+.5*x.fp.sum()+.5*x.fn.sum())
for c in ["lap_var","hp_std"]:
    q=img[c].quantile([1/3,2/3]).values
    ent["t"]=np.digitize(ent[c],q)
    print(c)
    for t,x in ent.groupby("t"):
        n_img=x.stem.nunique()
        print(f"  tercile {t}: imgs {n_img} PQ {pq(x):.3f} SQ {x.siou.sum()/x.tp.sum():.3f} FP/img-entry {x.fp.mean():.2f} FN/entry {x.fn.mean():.2f} TP/entry {x.tp.mean():.2f} | grp01/04 {pq(x[x.grp.isin(['01','04'])]):.3f} other {pq(x[~x.grp.isin(['01','04'])]):.3f} share01/04 {x.grp.isin(['01','04']).mean():.2f} nann {x.nann.mean():.2f}")
# bootstrap diff low vs high tercile lap_var
q=img.lap_var.quantile([1/3,2/3]).values; ent["t"]=np.digitize(ent.lap_var,q)
rng=np.random.default_rng(0); lo=ent[ent.t==0]; hi=ent[ent.t==2]
def bs(x):
    s=x.stem.unique(); return pq(pd.concat([x[x.stem==k] for k in rng.choice(s,len(s))]))
diffs=[bs(lo)-bs(hi) for _ in range(300)]
print("low-high lap_var PQ diff %.3f CI [%.3f, %.3f]"%(pq(lo)-pq(hi),*np.percentile(diffs,[2.5,97.5])))
# human agreement vs sharpness? use GT support rate in multi-ann images
G=G.merge(img[["stem","lap_var"]],on="stem"); G["t"]=np.digitize(G.lap_var,q)
print("multi-ann GT support rate by sharpness tercile:", G[G.nann>1].groupby("t").apply(lambda x:(x.sup>0).mean()).round(3).to_dict())
print("GT count per entry by tercile:", ent.groupby("t").apply(lambda x:(x.tp+x.fn).mean()).round(2).to_dict())
print("station x tercile counts:\n", pd.crosstab(img.st, np.digitize(img.lap_var,q)))
