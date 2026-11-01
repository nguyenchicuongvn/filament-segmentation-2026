import sys, pickle, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from fp_filter import _image_table
from filseg.data import load_train, with_maps
from filseg.dl import PostParams
ROOT=Path(__file__).resolve().parents[4]
P=ROOT/"outputs/colab/unet1024_effb4/probs_val"
stem="20110109104734Ch"
its=[i for i in with_maps(load_train(),P) if i.image_id==stem]
post=PostParams(**json.loads((ROOT/"configs/post_unet1024_effb4_hyst.json").read_text()))
t=_image_table((str(P),post,its[0].path,stem,[i.gt_rles for i in its]))
rec=pickle.load(open(Path(__file__).parent/"cands"/f"{stem}.pkl","rb"))
for a,b in zip(t[3],rec["base"]["ious"]): print(a.shape,b.shape,np.abs(a-b).max())
