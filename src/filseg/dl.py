"""Deep-learning pipeline: U-Net + instance post-processing.

The network predicts, at a reduced working resolution, a per-pixel filament
probability (channel 0) and optionally a "group" map (channel 1, targets
--targets group): every filament's mask dilated by a few pixels, minus the
pixels where two dilated filaments overlap. The group map teaches the network
which fragments annotators consider one filament, and where two filaments must
stay apart.

Instances are recovered from the maps by one of three modes (PostParams.mode):
  cc         threshold, drop small blobs, group fragments by a fixed dilation
  watershed  seeds = confident pixels; seeds closer than ~2*group_radius form one
             filament even across gaps in the foreground; each foreground pixel
             joins a seed group by watershed on the probability; foreground
             components without any seed are dropped (hysteresis-like)
fill_holes closes holes inside each instance (annotations never have holes).
  hyst       like cc, but the extent threshold (thresh) and the keep test are separate:
             a component of p > thresh is kept only if its peak exceeds seed_thresh
             (draw boundaries low, accept filaments only when confident)
  group      instances = connected regions of the predicted group map;
             foreground outside every region forms its own instances
Labels are finally upsampled to 2048x2048.
"""
from __future__ import annotations

import os
import zipfile
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


def _load_npz(path: Path) -> dict | None:
    """A cached .npz, or None if missing or unreadable (e.g. truncated by a killed run)."""
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    except (EOFError, OSError, ValueError, zipfile.BadZipFile):
        path.unlink(missing_ok=True)
        return None


def _save_npz(path: Path, **arrays) -> None:
    """Write atomically, so an interrupted run never leaves a partial cache file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


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
    if (z := _load_npz(path)) is not None:
        return z["img"], z["mask"], z["group"]
    img = cv2.resize(load_gray(item.path), (res, res), interpolation=cv2.INTER_AREA)
    m = cv2.resize(semantic_mask(item).astype(np.float32), (res, res), interpolation=cv2.INTER_AREA)
    mask = (m >= 0.5).astype(np.uint8)
    group = group_target(item, res)
    _save_npz(path, img=img, mask=mask, group=group)
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
    if (z := _load_npz(path)) is not None:
        return z["img"], z["soft"]
    gray = load_gray(entries[0].path)
    img = gray if res == W else cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA)
    acc = np.zeros((res, res), np.float32)
    for it in entries:
        m = semantic_mask(it).astype(np.float32)
        acc += m if res == W else cv2.resize(m, (res, res), interpolation=cv2.INTER_AREA)
    soft = np.round(acc / len(entries) * 255).astype(np.uint8)
    _save_npz(path, img=img, soft=soft)
    return img, soft


class SoftDataset(Dataset):
    """One sample per image with a soft multi-annotator target, optionally a random crop.

    With crop > 0, most crops are centred on a random filament pixel (fg_frac), the rest
    anywhere, so training sees native-resolution detail while inference runs on the whole image.
    """

    def __init__(self, groups: list[list[TrainImage]], res: int, augment: bool, crop: int = 0,
                 fg_frac: float = 0.7, instances: bool = False, inst_min_area: int = 10):
        self.groups, self.res, self.augment, self.crop, self.fg_frac = groups, res, augment, crop, fg_frac
        self.instances, self.inst_min_area = instances, inst_min_area

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
        x, y = torch.from_numpy(np.ascontiguousarray(img)[None]), torch.from_numpy(np.ascontiguousarray(mask))
        if not self.instances:
            return x, y
        # labels of the final (cropped, rotated, flipped) target, so they stay aligned with it
        lab = instance_labels(np.ascontiguousarray(mask[0]), self.inst_min_area)
        return x, y, torch.from_numpy(lab.astype(np.int64)[None])


def instance_labels(soft: np.ndarray, min_area: int) -> np.ndarray:
    """8-connected components of soft > 0 as int32 labels 1..n (background 0). Components smaller
    than min_area get -1: they still count as "another instance" in the blob loss of the others
    but have no blob term of their own."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((soft > 0).astype(np.uint8), connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < min_area
    small[0] = False
    valid = ~small
    valid[0] = False
    lut = np.zeros(n, np.int32)
    lut[valid] = np.arange(1, int(valid.sum()) + 1)
    lut[small] = -1
    return lut[lab]


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0) -> torch.Tensor:
    """BCE + dice_weight * soft Dice, Dice computed per sample and channel. With soft targets the Dice
    term pushes probabilities to 0/1 (maps stop meaning "fraction of annotators"); BCE alone is
    calibrated, so dice_weight < 1 trades overlap for calibration."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    if not dice_weight:
        return bce
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + target.sum((2, 3)) + 1)
    return bce + dice_weight * dice.mean()


def blob_dice_loss(logits: torch.Tensor, target: torch.Tensor, labels: torch.Tensor,
                   eps: float = 1.0) -> torch.Tensor:
    """Blob loss (instance-normalised soft Dice), channel 0, in closed form.

    For instance k of a sample, Dice is computed over the whole image with every other target
    instance (labels != 0 and != k, including the -1 small ones) masked out of both prediction and
    target, so a small filament weighs as much as a large one and a false blob in the background
    lowers every instance's Dice. The loss is the mean of 1 - Dice over all instances in the batch.
    """
    p = torch.sigmoid(logits[:, 0]).flatten(1)
    s = target[:, 0].flatten(1).to(p.dtype)
    lab = labels[:, 0].flatten(1).to(p.device).long()
    n = int(lab.max().item()) + 1 if lab.numel() else 0
    if n <= 1:
        return logits.sum() * 0
    idx = lab.clamp(min=0)  # small (-1) and background pixels go to bin 0, which is ignored
    zeros = p.new_zeros(p.shape[0], n)
    p_in = zeros.scatter_add(1, idx, p)
    a = zeros.scatter_add(1, idx, p * s)
    s_k = zeros.scatter_add(1, idx, s)
    count = zeros.scatter_add(1, idx, torch.ones_like(p))
    f = (p * (lab != 0)).sum(1, keepdim=True)
    pt = p.sum(1, keepdim=True)
    dice = (2 * a + eps) / (pt - f + p_in + s_k + eps)
    valid = count > 0
    valid[:, 0] = False
    if not valid.any():
        return logits.sum() * 0
    return (1 - dice[valid]).mean()


def false_component_loss(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5,
                         min_area: int = 10) -> torch.Tensor:
    """Penalty on predicted components that touch no target pixel (the predicted-component half of
    BiCC). Components of sigmoid > thresh (8-connected, area >= min_area, found on the detached
    map) with zero overlap with target > 0; loss = mean over them (pooled over the batch) of the
    mean probability inside each component."""
    p = torch.sigmoid(logits[:, 0])
    pred = (p.detach() > thresh).cpu().numpy().astype(np.uint8)
    tgt = (target[:, 0] > 0).cpu().numpy()
    terms = []
    for b in range(p.shape[0]):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(pred[b], connectivity=8)
        if n <= 1:
            continue
        area = stats[:, cv2.CC_STAT_AREA]
        hit = np.bincount(lab[tgt[b]], minlength=n) > 0
        keep = (area >= min_area) & ~hit
        keep[0] = False
        ks = np.nonzero(keep)[0]
        if not len(ks):
            continue
        lab_t = torch.from_numpy(lab.astype(np.int64)).to(p.device).flatten()
        sums = p.new_zeros(n).scatter_add(0, lab_t, p[b].flatten())
        ks_t = torch.from_numpy(ks).to(p.device)
        terms.append(sums[ks_t] / torch.from_numpy(area[ks].astype(np.float32)).to(p.device))
    if not terms:
        return logits.sum() * 0
    return torch.cat(terms).mean()


def train_loss(logits: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, w_blob: float,
               w_fpc: float, fpc_thresh: float = 0.5, min_area: int = 10) -> tuple[torch.Tensor, dict]:
    """bce_dice_loss + w_blob * blob_dice_loss + w_fpc * false_component_loss, and the three
    components (detached floats) for logging. A term with weight 0 is not computed."""
    base = bce_dice_loss(logits, target)
    blob = blob_dice_loss(logits, target, labels) if w_blob else logits.sum() * 0
    fpc = false_component_loss(logits, target, fpc_thresh, min_area) if w_fpc else logits.sum() * 0
    total = base + w_blob * blob + w_fpc * fpc
    return total, {"bce_dice": base.item(), "blob": blob.item(), "fpc": fpc.item()}


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
    group_line: int = 0        # cc/hyst: also dilate each fragment by a line of this half-length
                               # along its major axis, so fragments of one elongated filament merge
                               # along it but not sideways (research05: L 16, r 6 at 2048 px)
    out_dilate: int = 0
    mode: str = "cc"           # cc | hyst | watershed | group (see module docstring)
    seed_thresh: float = 0.7   # watershed: seed pixels
    group_thresh: float = 0.5  # group: threshold on the group map
    fill_holes: bool = False   # fill background enclosed by an instance
    # size-dependent extent (0 = off): on val, small instances come out ~25% larger than the
    # annotators' outlines and large ones ~13% smaller, so small ones are re-thresholded higher
    # and large ones grown into lower-probability pixels around them. Areas at map resolution.
    small_area: int = 0
    small_thresh: float = 0.0
    large_area: int = 0
    large_thresh: float = 0.0
    large_grow: int = 3

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


def _group_oriented(fg: np.ndarray, lab: np.ndarray, keep: np.ndarray, stats: np.ndarray,
                    length: int, radius: int) -> np.ndarray:
    """Group kept components after dilating each by disk(radius) union a line of half-length
    `length` along its major axis (orientation from the second moments)."""
    h_, w_ = fg.shape
    grouped = np.zeros_like(fg)
    pad = length + radius + 1
    for k in np.nonzero(keep)[0]:
        x, y, w, h = stats[k, :4]
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, w_), min(y + h + pad, h_)
        m = (lab[y0:y1, x0:x1] == k).astype(np.uint8)
        mo = cv2.moments(m, binaryImage=True)
        th = 0.5 * np.arctan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"])
        ker = np.zeros((2 * length + 1, 2 * length + 1), np.uint8)
        dx, dy = length * np.cos(th), length * np.sin(th)
        cv2.line(ker, (round(length - dx), round(length - dy)), (round(length + dx), round(length + dy)), 1, 3)
        if radius > 0:
            ker[length - radius:length + radius + 1, length - radius:length + radius + 1] |= _disk(radius)
        grouped[y0:y1, x0:x1] |= cv2.dilate(m, ker)
    _, glab = cv2.connectedComponents(grouped, connectivity=8)
    return np.where(fg > 0, glab, 0).astype(np.int32)


def instances_lowres(prob: np.ndarray, p: PostParams, group: np.ndarray | None = None) -> np.ndarray:
    """Instance labels at the probability map's resolution."""
    fg = (prob > p.thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= p.min_area
    if p.mode == "hyst":
        peak = np.zeros(n, np.float32)
        np.maximum.at(peak, lab.ravel(), prob.ravel())
        keep &= peak > p.seed_thresh
    keep[0] = False
    fg = keep[lab].astype(np.uint8)

    if p.mode == "watershed":
        from skimage.segmentation import watershed

        seeds = ((prob > p.seed_thresh) & (fg > 0)).astype(np.uint8)
        # group seeds by proximity (through background too), then label only the seed pixels
        grouped = cv2.dilate(seeds, _disk(p.group_radius)) if p.group_radius > 0 else seeds
        _, glab = cv2.connectedComponents(grouped, connectivity=8)
        markers = np.where(seeds > 0, glab, 0).astype(np.int32)
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
    elif p.group_line > 0:
        inst = _group_oriented(fg, lab, keep, stats, p.group_line, p.group_radius)
    else:
        grouped = cv2.dilate(fg, _disk(p.group_radius)) if p.group_radius > 0 else fg
        _, glab = cv2.connectedComponents(grouped, connectivity=8)
        inst = np.where(fg > 0, glab, 0).astype(np.int32)

    if p.out_dilate > 0:
        grown = cv2.dilate(inst.astype(np.float32), _disk(p.out_dilate))
        inst = np.where(inst > 0, inst, grown).astype(np.int32)
    if p.fill_holes:
        inst = fill_instance_holes(inst)
    if p.small_thresh or p.large_thresh:
        inst = adjust_extent(inst, prob, p)
    return _relabel(inst, p.min_area)


def adjust_extent(inst: np.ndarray, prob: np.ndarray, p: PostParams) -> np.ndarray:
    """Shrink small instances to pixels above small_thresh; grow large ones by up to large_grow px
    into unclaimed pixels above large_thresh."""
    out = inst.copy()
    areas = np.bincount(inst.ravel())
    if p.small_thresh and p.small_area:
        small = np.zeros(len(areas), bool)
        small[1:] = (areas[1:] > 0) & (areas[1:] < p.small_area)
        drop = small[inst] & (prob <= p.small_thresh)
        out[drop] = 0
    if p.large_thresh and p.large_area:
        large = np.zeros(len(areas), bool)
        large[1:] = areas[1:] >= p.large_area
        seed = np.where(large[inst], inst, 0).astype(np.float32)
        grown = cv2.dilate(seed, _disk(p.large_grow)).astype(np.int32)
        add = (out == 0) & (grown > 0) & (prob > p.large_thresh)
        out[add] = grown[add]
    return out


def fill_instance_holes(inst: np.ndarray) -> np.ndarray:
    """Background regions that do not reach the image border are holes; each is given to the
    instance around it (grown inward from its edge)."""
    bg = (inst == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(bg, connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    holes = (lab > 0) & ~np.isin(lab, border)
    if not holes.any():
        return inst
    out = inst.copy()
    k = np.ones((3, 3), np.uint8)
    for _ in range(200):
        todo = holes & (out == 0)
        if not todo.any():
            break
        grown = cv2.dilate(out.astype(np.float32), k).astype(np.int32)
        out = np.where(todo, grown, out)
    return out


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
