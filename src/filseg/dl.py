"""Deep-learning pipeline: U-Net + instance post-processing.

The network predicts, at a reduced working resolution, a per-pixel filament
probability (channel 0) and optionally a "group" map (channel 1, targets
--targets group): every filament's mask dilated by a few pixels, minus the
pixels where two dilated filaments overlap. The group map teaches the network
which fragments annotators consider one filament, and where two filaments must
stay apart.

Instances are recovered from the maps by one of three modes (PostParams.mode):
  cc         threshold, drop small blobs, group fragments by a fixed dilation
  watershed  seeds = confident pixels grouped by dilation; each foreground
             pixel joins a seed by watershed on the probability; foreground
             components without any seed are dropped (hysteresis-like)
  group      instances = connected regions of the predicted group map;
             foreground outside every region forms its own instances
Labels are finally upsampled to 2048x2048.
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


TARGET_CHANNELS = {"mask": 1, "group": 2, "soft": 1}


def build_model(encoder: str = "resnet34", pretrained: bool = True, classes: int = 1) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=1, classes=classes)


# ---------------------------------------------------------------- data


def semantic_mask(item: TrainImage) -> np.ndarray:
    """Union of all filament masks at full resolution (uint8 0/1)."""
    if not item.gt_rles:
        return np.zeros((H, W), np.uint8)
    return mask_util.decode(item.gt_rles).max(axis=2).astype(np.uint8)


def group_target(item: TrainImage, res: int) -> np.ndarray:
    """Each filament dilated by ~6 px at 1024 scale, minus pixels claimed by two or more."""
    r = max(1, round(6 * res / 1024))
    count = np.zeros((res, res), np.uint8)
    for rle in item.gt_rles:
        m = cv2.resize(mask_util.decode(rle).astype(np.float32), (res, res), interpolation=cv2.INTER_AREA)
        count += cv2.dilate((m >= 0.5).astype(np.uint8), _disk(r))
    return (count == 1).astype(np.uint8)


def cached(item: TrainImage, res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image, mask and group target resized to res x res, cached on disk as .npz."""
    path = CACHE_DIR / str(res) / f"{item.key}.v2.npz"
    if path.exists():
        z = np.load(path)
        return z["img"], z["mask"], z["group"]
    img = cv2.resize(load_gray(item.path), (res, res), interpolation=cv2.INTER_AREA)
    m = cv2.resize(semantic_mask(item).astype(np.float32), (res, res), interpolation=cv2.INTER_AREA)
    mask = (m >= 0.5).astype(np.uint8)
    group = group_target(item, res)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, img=img, mask=mask, group=group)
    return img, mask, group


class FilamentDataset(Dataset):
    def __init__(self, items: list[TrainImage], res: int, augment: bool, targets: str = "mask"):
        self.items, self.res, self.augment, self.targets = items, res, augment, targets

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        img, mask, group = cached(self.items[i], self.res)
        img = img.astype(np.float32) / 255.0
        chans = [mask] if self.targets == "mask" else [mask, group]
        mask = np.stack(chans).astype(np.float32)  # (C, H, W)
        if self.augment:
            rng = np.random.default_rng()
            k = int(rng.integers(4))
            img, mask = np.rot90(img, k), np.rot90(mask, k, axes=(1, 2))
            if rng.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, :, ::-1]
            # intensity: gamma + contrast/brightness jitter (sites differ in exposure)
            img = np.clip(img, 0, 1) ** float(rng.uniform(0.7, 1.4))
            img = img * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-0.08, 0.08))
        img = np.ascontiguousarray(img)[None]
        mask = np.ascontiguousarray(mask)
        return torch.from_numpy(img), torch.from_numpy(mask)


def soft_cached(entries: list[TrainImage], res: int) -> tuple[np.ndarray, np.ndarray]:
    """Image and soft target for one image at res x res: each pixel's target is the fraction of
    the image's annotators whose filaments cover it (uint8, 0..255). Cached as .npz."""
    path = CACHE_DIR / str(res) / f"{entries[0].image_id}.soft.npz"
    if path.exists():
        z = np.load(path)
        return z["img"], z["soft"]
    gray = load_gray(entries[0].path)
    img = gray if res == W else cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA)
    acc = np.zeros((res, res), np.float32)
    for it in entries:
        m = semantic_mask(it).astype(np.float32)
        acc += m if res == W else cv2.resize(m, (res, res), interpolation=cv2.INTER_AREA)
    soft = np.round(acc / len(entries) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, img=img, soft=soft)
    return img, soft


class SoftDataset(Dataset):
    """One sample per image with a soft multi-annotator target, optionally a random crop.

    With crop > 0, most crops are centred on a random filament pixel (fg_frac), the rest
    anywhere, so training sees native-resolution detail while inference runs on the whole image.
    """

    def __init__(self, groups: list[list[TrainImage]], res: int, augment: bool, crop: int = 0,
                 fg_frac: float = 0.7):
        self.groups, self.res, self.augment, self.crop, self.fg_frac = groups, res, augment, crop, fg_frac

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i):
        img, soft = soft_cached(self.groups[i], self.res)
        rng = np.random.default_rng()
        if self.crop and self.crop < self.res:
            c = self.crop
            ys, xs = np.nonzero(soft) if rng.random() < self.fg_frac else ((), ())
            if len(ys):
                j = int(rng.integers(len(ys)))
                cy, cx = ys[j] + int(rng.integers(-c // 4, c // 4)), xs[j] + int(rng.integers(-c // 4, c // 4))
            else:
                cy, cx = int(rng.integers(c // 2, self.res - c // 2)), int(rng.integers(c // 2, self.res - c // 2))
            y0 = int(np.clip(cy - c // 2, 0, self.res - c))
            x0 = int(np.clip(cx - c // 2, 0, self.res - c))
            img, soft = img[y0:y0 + c, x0:x0 + c], soft[y0:y0 + c, x0:x0 + c]
        img = img.astype(np.float32) / 255.0
        mask = soft.astype(np.float32)[None] / 255.0
        if self.augment:
            k = int(rng.integers(4))
            img, mask = np.rot90(img, k), np.rot90(mask, k, axes=(1, 2))
            if rng.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, :, ::-1]
            img = np.clip(img, 0, 1) ** float(rng.uniform(0.7, 1.4))
            img = img * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-0.08, 0.08))
        return torch.from_numpy(np.ascontiguousarray(img)[None]), torch.from_numpy(np.ascontiguousarray(mask))


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + soft Dice, Dice computed per sample and channel."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + target.sum((2, 3)) + 1)
    return bce + dice.mean()


# ---------------------------------------------------------------- inference


@torch.no_grad()
def predict_prob(model: torch.nn.Module, gray: np.ndarray, res: int, dev: torch.device,
                 tta: bool = True) -> np.ndarray:
    """Per-channel probabilities, shape (C, res, res), for one full-size grayscale image.
    Channel 0 is the filament mask; channel 1 (if the model has it) the group map."""
    img = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = torch.from_numpy(img)[None, None].to(dev)
    if not tta:
        return torch.sigmoid(model(x))[0].float().cpu().numpy()
    acc = None
    for flip in (False, True):
        xf = torch.flip(x, (3,)) if flip else x
        for k in range(4):
            y = torch.sigmoid(model(torch.rot90(xf, k, (2, 3))))
            y = torch.rot90(y, -k, (2, 3))
            y = torch.flip(y, (3,)) if flip else y
            acc = y if acc is None else acc + y
    return (acc / 8)[0].float().cpu().numpy()


@dataclass
class PostParams:
    thresh: float = 0.5
    min_area: int = 40         # pixels at prediction resolution
    group_radius: int = 0      # merge fragments (or seeds) closer than ~2*radius
    out_dilate: int = 0
    mode: str = "cc"           # cc | watershed | group (see module docstring)
    seed_thresh: float = 0.7   # watershed: seed pixels
    group_thresh: float = 0.5  # group: threshold on the group map

    def to_dict(self) -> dict:
        return asdict(self)


def _disk(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def _relabel(inst: np.ndarray, min_area: int) -> np.ndarray:
    """Drop instances smaller than min_area and renumber 1..n."""
    if inst.max() == 0:
        return inst.astype(np.int32)
    areas = np.bincount(inst.ravel())
    keep = areas >= min_area
    keep[0] = False
    lut = np.zeros(len(areas), np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1)
    return lut[inst]


def instances_lowres(prob: np.ndarray, p: PostParams, group: np.ndarray | None = None) -> np.ndarray:
    """Instance labels at the probability map's resolution."""
    fg = (prob > p.thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= p.min_area
    keep[0] = False
    fg = keep[lab].astype(np.uint8)

    if p.mode == "watershed":
        from skimage.segmentation import watershed

        seeds = ((prob > p.seed_thresh) & (fg > 0)).astype(np.uint8)
        if p.group_radius > 0:
            seeds = cv2.dilate(seeds, _disk(p.group_radius)) & fg
        _, markers = cv2.connectedComponents(seeds, connectivity=8)
        inst = watershed(-prob, markers, mask=fg > 0).astype(np.int32)
    elif p.mode == "group":
        if group is None:
            raise ValueError("mode 'group' needs the group map (a --targets group model)")
        regions = (group > p.group_thresh).astype(np.uint8)
        _, glab = cv2.connectedComponents(regions, connectivity=8)
        inst = np.where(fg > 0, glab, 0).astype(np.int32)
        # foreground the group map does not cover: its own components
        rest = ((fg > 0) & (inst == 0)).astype(np.uint8)
        if rest.any():
            _, rlab = cv2.connectedComponents(rest, connectivity=8)
            inst = np.where(rlab > 0, rlab + inst.max(), inst).astype(np.int32)
    else:
        grouped = cv2.dilate(fg, _disk(p.group_radius)) if p.group_radius > 0 else fg
        _, glab = cv2.connectedComponents(grouped, connectivity=8)
        inst = np.where(fg > 0, glab, 0).astype(np.int32)

    if p.out_dilate > 0:
        grown = cv2.dilate(inst.astype(np.float32), _disk(p.out_dilate))
        inst = np.where(inst > 0, inst, grown).astype(np.int32)
    return _relabel(inst, p.min_area)


def upsample_labels(inst: np.ndarray, out_size: int = W) -> np.ndarray:
    return cv2.resize(inst.astype(np.float32), (out_size, out_size),
                      interpolation=cv2.INTER_NEAREST).astype(np.int32)


def instances(prob: np.ndarray, p: PostParams, out_size: int = W,
              group: np.ndarray | None = None) -> np.ndarray:
    """Instance label image at out_size x out_size from a probability map."""
    return upsample_labels(instances_lowres(prob, p, group), out_size)


def load_maps(probs_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Cached uint8 maps written by scripts/infer_probs.py -> float (mask, group or None)."""
    prob = np.load(Path(probs_dir) / f"{stem}.npy").astype(np.float32) / 255.0
    g = Path(probs_dir) / "group" / f"{stem}.npy"
    group = np.load(g).astype(np.float32) / 255.0 if g.exists() else None
    return prob, group
