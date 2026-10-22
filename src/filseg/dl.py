"""Deep-learning pipeline: semantic U-Net + connected-component instances.

The network predicts a per-pixel filament probability at a reduced working
resolution. Instances are recovered by thresholding, removing small blobs,
grouping nearby fragments (dilation), and upsampling labels to 2048x2048.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util
from torch.utils.data import Dataset

from .data import H, W, TrainImage, load_gray

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(encoder: str = "resnet34", pretrained: bool = True) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=1, classes=1)


# ---------------------------------------------------------------- data


def semantic_mask(item: TrainImage) -> np.ndarray:
    """Union of all filament masks at full resolution (uint8 0/1)."""
    if not item.gt_rles:
        return np.zeros((H, W), np.uint8)
    return mask_util.decode(item.gt_rles).max(axis=2).astype(np.uint8)


def cached(item: TrainImage, res: int) -> tuple[np.ndarray, np.ndarray]:
    """Image and mask resized to res x res, cached on disk as .npz."""
    path = CACHE_DIR / str(res) / f"{item.key}.npz"
    if path.exists():
        z = np.load(path)
        return z["img"], z["mask"]
    img = cv2.resize(load_gray(item.path), (res, res), interpolation=cv2.INTER_AREA)
    m = cv2.resize(semantic_mask(item).astype(np.float32), (res, res), interpolation=cv2.INTER_AREA)
    mask = (m >= 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, img=img, mask=mask)
    return img, mask


class FilamentDataset(Dataset):
    def __init__(self, items: list[TrainImage], res: int, augment: bool):
        self.items, self.res, self.augment = items, res, augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        img, mask = cached(self.items[i], self.res)
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)
        if self.augment:
            rng = np.random.default_rng()
            k = int(rng.integers(4))
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            if rng.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            # intensity: gamma + contrast/brightness jitter (sites differ in exposure)
            img = np.clip(img, 0, 1) ** float(rng.uniform(0.7, 1.4))
            img = img * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-0.08, 0.08))
        img = np.ascontiguousarray(img)[None]
        mask = np.ascontiguousarray(mask)[None]
        return torch.from_numpy(img), torch.from_numpy(mask)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


# ---------------------------------------------------------------- inference


@torch.no_grad()
def predict_prob(model: torch.nn.Module, gray: np.ndarray, res: int, dev: torch.device,
                 tta: bool = True) -> np.ndarray:
    """Filament probability at res x res for one full-size grayscale image."""
    img = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = torch.from_numpy(img)[None, None].to(dev)
    if not tta:
        return torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    acc = torch.zeros_like(x)
    for flip in (False, True):
        xf = torch.flip(x, (3,)) if flip else x
        for k in range(4):
            y = torch.sigmoid(model(torch.rot90(xf, k, (2, 3))))
            y = torch.rot90(y, -k, (2, 3))
            acc += torch.flip(y, (3,)) if flip else y
    return (acc / 8)[0, 0].float().cpu().numpy()


@dataclass
class PostParams:
    thresh: float = 0.5
    min_area: int = 40         # pixels at prediction resolution
    group_radius: int = 0      # merge fragments closer than ~2*radius
    out_dilate: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _disk(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def instances(prob: np.ndarray, p: PostParams, out_size: int = W) -> np.ndarray:
    """Instance label image at out_size x out_size from a probability map."""
    fg = (prob > p.thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= p.min_area
    keep[0] = False
    fg = keep[lab].astype(np.uint8)
    grouped = cv2.dilate(fg, _disk(p.group_radius)) if p.group_radius > 0 else fg
    _, glab = cv2.connectedComponents(grouped, connectivity=8)
    inst = np.where(fg > 0, glab, 0).astype(np.int32)
    if p.out_dilate > 0:
        grown = cv2.dilate(inst.astype(np.float32), _disk(p.out_dilate))
        inst = np.where(inst > 0, inst, grown).astype(np.int32)
    ids = np.unique(inst)
    ids = ids[ids > 0]
    lut = np.zeros(int(inst.max()) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    inst = lut[inst]
    return cv2.resize(inst.astype(np.float32), (out_size, out_size),
                      interpolation=cv2.INTER_NEAREST).astype(np.int32)
