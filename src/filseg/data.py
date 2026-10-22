"""Dataset access: image paths, COCO annotations, and RLE helpers."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"
TRAIN_IMG_DIR = DATA_DIR / "train" / "train_images"
TEST_IMG_DIR = DATA_DIR / "test" / "test_images"
TRAIN_ANN = DATA_DIR / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

H = W = 2048


@dataclass
class TrainImage:
    """One annotator's labelling of one image. Many images have 2-3 annotators,
    each a separate entry (the host scores every annotator-image separately)."""
    key: str               # annotator-image id, e.g. 040301-20140609195854Bh
    image_id: str          # file stem, e.g. 20140609195854Bh
    path: Path
    gt_rles: list[dict]    # one compressed COCO RLE per filament


def load_gray(path: Path) -> np.ndarray:
    """Images are stored as 3 identical channels; return a single uint8 channel."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def polygon_to_rle(polygons: list[list[float]], h: int = H, w: int = W) -> dict:
    rles = mask_util.frPyObjects(polygons, h, w)
    return mask_util.merge(rles)


def load_train() -> list[TrainImage]:
    coco = json.loads(TRAIN_ANN.read_text())
    by_image: dict[str, list[dict]] = {im["id"]: [] for im in coco["images"]}
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(polygon_to_rle(ann["segmentation"]))
    out = []
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        out.append(TrainImage(im["id"], stem, TRAIN_IMG_DIR / im["file_name"], by_image[im["id"]]))
    out.sort(key=lambda t: (t.image_id, t.key))
    return out


def test_paths() -> list[Path]:
    return sorted(TEST_IMG_DIR.glob("*.jpeg"))


def encode_masks(labels: np.ndarray) -> list[dict]:
    """Encode an instance label image (0 = background) into one RLE per instance."""
    rles = []
    for k in range(1, int(labels.max()) + 1):
        m = labels == k
        if m.any():
            rles.append(mask_util.encode(np.asfortranarray(m.astype(np.uint8))))
    return rles


def rle_counts_str(rle: dict) -> str:
    c = rle["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def split(items: list[TrainImage], which: str, seed: int = 42) -> list[TrainImage]:
    """Fixed 70/30 split by image (all annotators of an image land on the same side);
    'val' is held out from all tuning and training."""
    if which == "all":
        return list(items)
    stems = sorted({t.image_id for t in items})
    random.Random(seed).shuffle(stems)
    cut = int(0.7 * len(stems))
    chosen = set(stems[:cut] if which == "train" else stems[cut:])
    return [t for t in items if t.image_id in chosen]
