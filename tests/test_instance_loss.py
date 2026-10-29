"""Recipe v2 (docs/PLAN.md N3): blob loss, false-component loss, instance labels, EMA plumbing."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import filseg.dl as dl  # noqa: E402
from filseg.dl import (SoftDataset, blob_dice_loss, build_model, false_component_loss,  # noqa: E402
                       instance_labels, train_loss)


def _train_unet():
    spec = importlib.util.spec_from_file_location("train_unet", ROOT / "scripts" / "train_unet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _soft_with_instances(size=48, seed=0):
    """Soft target with 4 components: three normal ones and one of 4 px (below min_area 10)."""
    rng = np.random.default_rng(seed)
    s = np.zeros((size, size), np.float32)
    s[2:10, 3:15] = rng.uniform(0.3, 1.0, (8, 12))
    s[20:30, 20:24] = rng.uniform(0.3, 1.0, (10, 4))
    s[35:44, 5:12] = rng.uniform(0.3, 1.0, (9, 7))
    s[40:42, 40:42] = 0.5  # 4 px: label -1
    return s


def _brute_blob(logits, target, labels, eps=1.0):
    """Blob loss by building each instance's masked prediction and target explicitly."""
    p = torch.sigmoid(logits[:, 0])
    s = target[:, 0]
    lab = labels[:, 0]
    terms = []
    for b in range(p.shape[0]):
        for k in torch.unique(lab[b]).tolist():
            if k < 1:
                continue
            keep = (lab[b] == 0) | (lab[b] == k)
            pk = p[b] * keep
            sk = s[b] * (lab[b] == k)
            dice = (2 * (pk * sk).sum() + eps) / (pk.sum() + sk.sum() + eps)
            terms.append(1 - dice)
    return torch.stack(terms).mean()


def test_instance_labels_small_component_is_minus_one():
    lab = instance_labels(_soft_with_instances(), min_area=10)
    assert lab.dtype == np.int32
    assert set(np.unique(lab).tolist()) == {-1, 0, 1, 2, 3}
    assert (lab[40:42, 40:42] == -1).all()
    assert ((lab != 0) == (_soft_with_instances() > 0)).all()


def test_blob_loss_matches_brute_force():
    torch.manual_seed(0)
    soft = np.stack([_soft_with_instances(seed=0), _soft_with_instances(seed=1)[::-1].copy()])
    labels = torch.from_numpy(np.stack([instance_labels(s, 10) for s in soft]).astype(np.int64)[:, None])
    target = torch.from_numpy(soft[:, None])
    logits_a = torch.randn(2, 1, 48, 48, dtype=torch.float64, requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    fast = blob_dice_loss(logits_a, target.double(), labels)
    slow = _brute_blob(logits_b, target.double(), labels)
    assert abs(fast.item() - slow.item()) < 1e-5
    fast.backward()
    slow.backward()
    assert torch.allclose(logits_a.grad, logits_b.grad, atol=1e-5)
    # float32 (the training dtype) agrees too
    l32 = blob_dice_loss(logits_a.detach().float(), target, labels)
    assert abs(l32.item() - slow.item()) < 1e-5


def test_blob_loss_no_instances():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.zeros(2, 1, 16, 16)
    for labels in (torch.zeros(2, 1, 16, 16, dtype=torch.int64),
                   torch.full((2, 1, 16, 16), -1, dtype=torch.int64)):  # only small components
        loss = blob_dice_loss(logits, target, labels)
        assert loss.item() == 0 and torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_false_component_loss():
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 2:8, 2:8] = 0.6
    logits = torch.full((1, 1, 32, 32), -6.0)
    logits[0, 0, 4:10, 4:10] = 2.0   # overlaps the target: not penalised
    logits[0, 0, 20:26, 18:25] = torch.linspace(0.5, 3.0, 7)  # isolated blob of 42 px
    logits[0, 0, 28:30, 2:4] = 3.0   # isolated but 4 px < min_area
    logits.requires_grad_(True)
    loss = false_component_loss(logits, target, thresh=0.5, min_area=10)
    expected = torch.sigmoid(logits.detach()[0, 0, 20:26, 18:25]).mean()
    assert abs(loss.item() - expected.item()) < 1e-6
    loss.backward()
    g = logits.grad[0, 0]
    assert (g[20:26, 18:25] > 0).all() and g[4:10, 4:10].abs().sum() == 0
    # without the isolated blobs: 0, finite, backward works
    logits2 = logits.detach().clone()
    logits2[0, 0, 20:30] = -6.0
    logits2.requires_grad_(True)
    loss2 = false_component_loss(logits2, target, 0.5, 10)
    assert loss2.item() == 0
    loss2.backward()
    assert torch.isfinite(logits2.grad).all()
    # nothing predicted at all
    assert false_component_loss(torch.full((2, 1, 8, 8), -5.0), torch.zeros(2, 1, 8, 8)).item() == 0


def test_train_loss_components():
    soft = _soft_with_instances()
    labels = torch.from_numpy(instance_labels(soft, 10).astype(np.int64))[None, None]
    target = torch.from_numpy(soft)[None, None]
    logits = torch.randn(1, 1, 48, 48, requires_grad=True)
    total, comp = train_loss(logits, target, labels, 1.0, 0.3, 0.5, 10)
    base = dl.bce_dice_loss(logits, target).item()
    assert abs(total.item() - (base + comp["blob"] + 0.3 * comp["fpc"])) < 1e-5
    assert abs(comp["bce_dice"] - base) < 1e-6
    total.backward()
    assert torch.isfinite(logits.grad).all()
    total0, comp0 = train_loss(logits, target, labels, 0.0, 0.0)
    assert comp0["blob"] == 0 and comp0["fpc"] == 0 and abs(total0.item() - base) < 1e-6


@pytest.mark.parametrize("crop", [0, 24])
def test_soft_dataset_labels_aligned(monkeypatch, crop):
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (48, 48), dtype=np.uint8)
    soft = (_soft_with_instances() * 255).astype(np.uint8)
    monkeypatch.setattr(dl, "soft_cached", lambda entries, res: (img, soft))
    ds = SoftDataset([[None]], 48, augment=True, crop=crop, instances=True, inst_min_area=10)
    for _ in range(40):
        x, y, lab = ds[0]
        assert lab.dtype == torch.int64 and lab.shape == y.shape == x.shape
        assert ((lab != 0) == (y > 0)).all()
    plain = SoftDataset([[None]], 48, augment=True, crop=crop)[0]
    assert len(plain) == 2


def test_default_args_are_plain_recipe():
    tu = _train_unet()
    args = tu.parse_args(["--targets", "soft"])
    assert args.blob_weight == 0 and args.fpc_weight == 0 and args.ema == 0
    assert not tu.instance_loss_on(args)
    model = torch.nn.Linear(2, 2)
    assert tu.setup_ema(model, args.ema) is None
    meta = tu.run_meta(args)
    assert not set(meta) & set(tu.V2_DEFAULTS)
    assert tu.best_state(model, None, meta).keys() == {"model", "meta"}
    with pytest.raises(SystemExit):
        tu.parse_args(["--blob-weight", "1.0"])  # needs --targets soft
    v2 = tu.parse_args(["--targets", "soft", "--res", "2048", "--blob-weight", "1", "--fpc-weight", "0.3",
                        "--ema", "0.998"])
    assert tu.instance_loss_on(v2) and tu.resolve_inst_min_area(v2) == 40
    assert tu.run_meta(v2)["inst_min_area"] == 40 and tu.run_meta(v2)["ema"] == 0.998


def test_ema_checkpoint_loads_into_build_model():
    tu = _train_unet()
    torch.manual_seed(0)
    model = build_model("resnet18", pretrained=False)
    ema = tu.setup_ema(model, 0.9)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 1, 64, 64)
    for _ in range(3):
        loss = model(x).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update_parameters(model)
    state = tu.best_state(model, ema, {"encoder": "resnet18"})
    assert {"model", "model_raw", "meta"} == state.keys()
    fresh = build_model("resnet18", pretrained=False)
    fresh.load_state_dict(state["model"])
    fresh.eval()
    ema.module.eval()
    with torch.no_grad():
        assert torch.allclose(fresh(x), ema.module(x), atol=1e-6)
        model.eval()
        assert not torch.allclose(fresh(x), model(x), atol=1e-4)  # EMA differs from the raw weights
    # resume: the EMA state round-trips through state_dict
    ema2 = tu.setup_ema(build_model("resnet18", pretrained=False), 0.9)
    ema2.load_state_dict(ema.state_dict())
    ema2.eval()
    with torch.no_grad():
        assert torch.allclose(ema2.module(x), ema.module(x), atol=1e-6)
