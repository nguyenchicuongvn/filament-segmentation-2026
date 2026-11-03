import pickle, numpy as np, pandas as pd
from pathlib import Path
H=Path(__file__).parent
d=pickle.load(open(H/"tables.pkl","rb")); P,G,ent=d["P"],d["G"],d["ent"]
st=pd.read_csv(H/"station_stats.csv").set_index("stem"); q=np.quantile(st[st.split=="train"].hp,[1/3,2/3])
for X in (P,G,ent): X["t"]=[np.digitize(st.loc[s,"hp"],q) for s in X.stem]
G["sz"]=pd.cut(G.area2048,[0,800,3200,1e9],labels=["<800","800-3200",">3200"])
print("GT recall by hp tercile x size (n):")
print(G.groupby(["t","sz"]).det.agg(["mean","size"]).unstack().round(2))
print("GT recall, multi-ann, supported vs lone, by tercile:")
m=G[G.nann>1]; print(m.groupby(["t",m.sup>0]).det.mean().unstack().round(3))
print("share of GTs < 800px by tercile:", G.groupby("t").apply(lambda x:(x.area2048<800).mean()).round(3).to_dict())
P["bad"]=~P.good if "good" in P else P.miou<0.205
print("pred count/img, bad share, median area by tercile:")
print(P.groupby("t").agg(n=("stem","size"),imgs=("stem","nunique"),bad=("bad","mean"),area=("area","median")).round(3))
print("SQ by tercile:", ent.groupby("t").apply(lambda x:x.siou.sum()/x.tp.sum()).round(3).to_dict())
