"""Scale / photometric TTA probe on E9a holdout (MPS). For each val image: D4 TTA at 1024 (should
match the cached map), plus maps at input scale 0.875 and 1.125 (resize 2048 -> 896 / 1152, D4,
resize prob back to 1024), plus perturbed inputs (gaussian blur sigma 1.5 px @1024 = 'bad seeing',
gamma 0.8). Saves uint8 maps per variant in out/<variant>/<stem>.npy.
"""
import sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from filseg.data import load_gray, load_train, split  # noqa
from filseg.dl import device  # noqa
from infer_probs import load_model  # noqa

OUT = Path(__file__).parent / "tta"
VARIANTS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["s896", "s1152", "blur", "base"]


@torch.no_grad()
def d4(model, img, dev):
    x = torch.from_numpy(img)[None, None].to(dev)
    acc = 0
    for flip in (False, True):
        xf = torch.flip(x, (3,)) if flip else x
        for k in range(4):
            y = torch.sigmoid(model(torch.rot90(xf, k, (2, 3))))
            y = torch.rot90(y, -k, (2, 3))
            acc = acc + (torch.flip(y, (3,)) if flip else y)
    return (acc / 8)[0, 0].float().cpu().numpy()


def main():
    dev = device()
    models, res, _ = load_model([ROOT / "outputs/colab/unet1024_effb4/unet1024_effb4.pt"], dev)
    m = models[0]
    paths = sorted({t.image_id: t.path for t in split(load_train(), "val")}.items())
    for v in VARIANTS:
        (OUT / v).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, (stem, path) in enumerate(paths):
        gray = load_gray(path)
        for v in VARIANTS:
            f = OUT / v / f"{stem}.npy"
            if f.exists():
                continue
            s = {"s896": 896, "s1152": 1152}.get(v, 1024)
            img = cv2.resize(gray, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            if v == "blur":
                img = cv2.GaussianBlur(img, (0, 0), 1.5)
            p = d4(m, img, dev)
            if s != 1024:
                p = cv2.resize(p, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            np.save(f, np.round(p * 255).astype(np.uint8))
        if i % 20 == 0:
            print(i, f"{time.time() - t0:.0f}s", flush=True)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
