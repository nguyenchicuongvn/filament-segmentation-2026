"""Mask R-CNN instance member (PLAN card N4): data, model, inference and the two ways of turning
detections into disjoint instances.

- ``detections_to_labels``: the detector alone (instances painted in score order).
- ``fuse_with_unet``: the U-Net map keeps the pixels and the detector decides the grouping: a U-Net
  component claimed by several detections is split between them, U-Net components covered by one
  detection are merged, and confident detections the U-Net missed can be added.

Detections are cached per image as ``<dir>/<stem>.npz`` (scores float32 (n,), masks uint8 (n, res,
res) = mask probability x 255), so the network runs once and the post-processing is tuned on CPU.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.utils.data import Dataset

from .data import TrainImage, load_gray
from .dl import CACHE_DIR, _load_npz, _save_npz, _relabel


# ---------------------------------------------------------------- data

def instance_cached(item: TrainImage, res: int) -> tuple[np.ndarray, np.ndarray]:
    """Image and this annotator's instance label image (int16, 0 = background) at res x res."""
    path = CACHE_DIR / str(res) / f"{item.key}.inst.npz"
    if (z := _load_npz(path)) is not None:
        return z["img"], z["labels"]
    img = cv2.resize(load_gray(item.path), (res, res), interpolation=cv2.INTER_AREA)
    labels = np.zeros((res, res), np.int16)
    for k, rle in enumerate(item.gt_rles, 1):
        m = cv2.resize(mask_util.decode(rle).astype(np.float32), (res, res), interpolation=cv2.INTER_AREA)
        labels[m >= 0.5] = k  # an annotator's own masks hardly overlap; the later one wins
    _save_npz(path, img=img, labels=labels)
    return img, labels


def targets_from_labels(labels: np.ndarray, min_area: int = 4) -> dict:
    """torchvision detection targets (boxes xyxy, labels = 1, masks) from an instance label image."""
    ids = [k for k in np.unique(labels) if k > 0]
    masks, boxes = [], []
    for k in ids:
        m = labels == k
        if m.sum() < min_area:
            continue
        ys, xs = np.nonzero(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        masks.append(m)
    n = len(masks)
    res = labels.shape
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(n, 4),
            "labels": torch.ones(n, dtype=torch.int64),
            "masks": torch.from_numpy(np.stack(masks).astype(np.uint8)) if n
            else torch.zeros((0, *res), dtype=torch.uint8)}


class DetectionDataset(Dataset):
    """One sample per annotator entry (the metric scores every entry): image + that annotator's
    instances, with the same rot90 / flip / intensity augmentation as the U-Net."""

    def __init__(self, items: list[TrainImage], res: int, augment: bool):
        self.items, self.res, self.augment = items, res, augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        img, labels = instance_cached(self.items[i], self.res)
        img = img.astype(np.float32) / 255.0
        if self.augment:
            rng = np.random.default_rng()
            k = int(rng.integers(4))
            img, labels = np.rot90(img, k), np.rot90(labels, k)
            if rng.random() < 0.5:
                img, labels = img[:, ::-1], labels[:, ::-1]
            img = np.clip(img, 0, 1) ** float(rng.uniform(0.7, 1.4))
            img = np.clip(img * float(rng.uniform(0.8, 1.2)) + float(rng.uniform(-0.08, 0.08)), 0, 1)
        img = torch.from_numpy(np.ascontiguousarray(img))[None].expand(3, -1, -1).contiguous()
        return img, targets_from_labels(np.ascontiguousarray(labels))


def collate(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------- model

def build_detector(res: int, pretrained: bool = True, max_dets: int = 100) -> torch.nn.Module:
    """torchvision Mask R-CNN (ResNet-50 FPN v2, COCO weights, BSD-3) with a 1-class head, working at
    the cached resolution (no internal resizing)."""
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights, min_size=res, max_size=res,
                                     box_detections_per_img=max_dets, box_score_thresh=0.01)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    return model


@torch.no_grad()
def predict_detections(model: torch.nn.Module, gray: np.ndarray, res: int, dev: torch.device,
                       min_score: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Scores (n,) and mask probabilities (n, res, res) uint8 of one full-resolution image."""
    model.eval()
    img = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = torch.from_numpy(img)[None].expand(3, -1, -1).contiguous().to(dev)
    out = model([x])[0]
    keep = out["scores"] >= min_score
    scores = out["scores"][keep].float().cpu().numpy()
    masks = (out["masks"][keep, 0].float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    return scores, masks


def save_detections(path: Path, scores: np.ndarray, masks: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, scores=scores.astype(np.float32), masks=masks.astype(np.uint8))


def load_detections(dets_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(dets_dir) / f"{stem}.npz") as z:
        return z["scores"], z["masks"]


# ---------------------------------------------------------------- detections -> instances

@dataclass
class DetParams:
    score_thresh: float = 0.5
    mask_thresh: float = 0.5
    min_visible: float = 0.5   # drop a detection if less than this share of it is left after painting
    min_area: int = 50         # at 1024 px

    def to_dict(self) -> dict:
        return asdict(self)


def detections_to_labels(scores: np.ndarray, masks: np.ndarray, p: DetParams) -> np.ndarray:
    """Disjoint instances from detections: paint in descending score order, a pixel belongs to the
    first detection that claims it; detections mostly hidden by better ones are dropped."""
    res = masks.shape[1:] if len(masks) else (1024, 1024)
    labels = np.zeros(res, np.int32)
    k = 0
    for i in np.argsort(-scores):
        if scores[i] < p.score_thresh:
            break
        m = masks[i] >= p.mask_thresh * 255
        area = int(m.sum())
        if area < p.min_area:
            continue
        free = m & (labels == 0)
        if free.sum() < p.min_visible * area:
            continue
        k += 1
        labels[free] = k
    return _relabel(labels, p.min_area)


@dataclass
class FuseParams:
    score_thresh: float = 0.5   # detections that take part in the grouping
    claim: float = 0.3          # a detection claims a U-Net component if this share of the detection lies in it
    merge: bool = True          # merge U-Net components that one detection covers
    add_thresh: float = 1.01    # add detections the U-Net missed if their score >= this (> 1 = never)
    require_det: bool = False   # drop U-Net components no detection touches
    min_area: int = 50

    def to_dict(self) -> dict:
        return asdict(self)


def fuse_with_unet(inst: np.ndarray, scores: np.ndarray, masks: np.ndarray, p: FuseParams) -> np.ndarray:
    """Regroup U-Net instances (label image at the detection resolution) with the detections.

    The U-Net decides which pixels are filament; the detector decides how they are grouped:
    1. merge: U-Net components that each lie mostly (>= 50%) inside the same detection become one;
    2. split: a component claimed by two or more detections is divided between them, every pixel
       going to the claiming detection with the highest mask probability;
    3. add / require_det as in FuseParams. The result is a disjoint label image."""
    assert inst.shape == masks.shape[1:] or not len(masks), (inst.shape, masks.shape)
    dets = [(s, m) for s, m in zip(scores, masks) if s >= p.score_thresh]
    dets.sort(key=lambda t: -t[0])
    out = inst.astype(np.int32).copy()
    n = int(out.max())
    if not dets:
        return out if not p.require_det else np.zeros_like(out)
    probs = np.stack([m for _, m in dets]).astype(np.float32) / 255.0   # (d, H, W)
    binm = probs >= 0.5
    comp_ids = [k for k in range(1, n + 1)]
    area = np.bincount(out.ravel(), minlength=n + 1)
    # overlap[d, k] = pixels of detection d inside component k
    overlap = np.zeros((len(dets), n + 1), np.int64)
    for d in range(len(dets)):
        overlap[d] = np.bincount(out[binm[d]], minlength=n + 1)
    det_area = binm.reshape(len(dets), -1).sum(1)
    # 1. merge components covered by one detection
    if p.merge:
        parent = np.arange(n + 1)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for d in range(len(dets)):
            inside = [k for k in comp_ids if area[k] and overlap[d, k] >= 0.5 * area[k]]
            for k in inside[1:]:
                parent[find(k)] = find(inside[0])
        roots = np.array([find(k) for k in range(n + 1)])
        out = roots[out]
        out[inst == 0] = 0
    # 2. split components claimed by several detections
    nxt = int(out.max()) + 1
    for k in np.unique(out):
        if k == 0:
            continue
        comp = out == k
        claim = [d for d in range(len(dets))
                 if det_area[d] and (binm[d] & comp).sum() >= p.claim * det_area[d]]
        if len(claim) < 2:
            continue
        owner = np.argmax(probs[claim][:, comp], axis=0)   # index into claim, per component pixel
        ys, xs = np.nonzero(comp)
        for j in range(1, len(claim)):
            sel = owner == j
            out[ys[sel], xs[sel]] = nxt
            nxt += 1
    # 3. detections no U-Net pixel supports
    touched = np.zeros(len(dets), bool)
    for d in range(len(dets)):
        touched[d] = (out[binm[d]] > 0).any()
    if p.require_det:
        keep_ids = set()
        for d in range(len(dets)):
            keep_ids.update(np.unique(out[binm[d]]).tolist())
        keep_ids.discard(0)
        out[~np.isin(out, list(keep_ids))] = 0
    if p.add_thresh <= 1:
        for d in range(len(dets)):
            if not touched[d] and dets[d][0] >= p.add_thresh:
                free = binm[d] & (out == 0)
                out[free] = nxt
                nxt += 1
    return _relabel(out, p.min_area)
