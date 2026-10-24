"""Watch for Colab runs on Kaggle and submit each one as soon as it lands.

For every run in --runs, polls the private dataset <owner>/filseg-<run>. When it
is ready:
  1. download to outputs/colab/<run> (inner probs_*.zip archives unpacked)
  2. tune post-processing on its val maps (cc mode; also group mode if it has a group map)
  3. train the instance FP filter on its val maps (5-fold CV)
  4. build the test submission with the filter and submit it
Once two or more runs are in, the mean of their 1024 px maps is tuned the same
way and submitted when its val PQ beats every single run.

Progress is kept in outputs/colab/watch_state.json, so the watcher can be
restarted; a submission refused by the daily limit is retried on later polls.

Usage:
  python scripts/watch_colab.py --runs unet1024_r34 unet1024_r34_grp unet1024_effb4 unet1024_cnxt unet1024_r34_e60
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
COMPETITION = "filament-segmentation-2026"
STATE = ROOT / "outputs" / "colab" / "watch_state.json"


def log(msg: str) -> None:
    print(f"[{datetime.now():%m-%d %H:%M}] {msg}", flush=True)


def sh(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    log(">> " + " ".join(map(str, cmd)))
    r = subprocess.run(list(map(str, cmd)), cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd[1] if len(cmd) > 1 else cmd[0]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"runs": {}, "ensemble": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2))


def slug(run: str) -> str:
    return "filseg-" + run.replace("_", "-")


def dataset_ready(owner: str, run: str) -> bool:
    r = subprocess.run(["kaggle", "datasets", "status", f"{owner}/{slug(run)}"], capture_output=True, text=True)
    return r.returncode == 0 and "ready" in r.stdout.lower()


def download(owner: str, run: str) -> Path:
    out = ROOT / "outputs" / "colab" / run
    sh(["kaggle", "datasets", "download", f"{owner}/{slug(run)}", "-p", out, "--unzip", "-q"])
    for z in list(out.glob("*.zip")):  # --dir-mode zip uploads sub-directories as archives
        with zipfile.ZipFile(z) as f:
            f.extractall(out / z.stem)
        z.unlink()
    # an archive may contain its own top directory: flatten probs_x/probs_x -> probs_x
    for d in out.glob("probs_*"):
        inner = d / d.name
        if inner.is_dir():
            for f in inner.iterdir():
                f.rename(d / f.name)
            inner.rmdir()
    return out


def best_line(log_text: str) -> float:
    """First result line of tune_post.py output -> its pq."""
    for line in log_text.splitlines():
        if line.startswith("{'pq'"):
            return float(line.split("'pq': ")[1].split(",")[0])
    raise RuntimeError("no result line in tune_post output")


def tune_and_filter(name: str, probs_val: Path, modes: list[str]) -> dict:
    """Tune post-processing (best of the given modes), then the FP filter. Returns a summary."""
    best = None
    for mode in modes:
        cfg = ROOT / "configs" / f"post_{name}{'' if mode == 'cc' else '_' + mode}.json"
        r = sh([PY, "scripts/tune_post.py", "--probs", probs_val, "--mode", mode, "--out", cfg])
        (ROOT / "docs" / "logs" / f"tune_{name}_{mode}.log").write_text(r.stdout)
        pq = best_line(r.stdout)
        log(f"{name}: tuned {mode} -> val pq {pq:.4f}")
        if best is None or pq > best[0]:
            best = (pq, cfg, mode)
    filt = ROOT / "configs" / f"fpfilter_{name}.pkl"
    r = sh([PY, "scripts/fp_filter.py", "--probs", probs_val, "--post", best[1], "--out", filt])
    (ROOT / "docs" / "logs" / f"fpfilter_{name}.log").write_text(r.stdout)
    f = pickle.loads(filt.read_bytes())
    log(f"{name}: FP filter val pq {f['base_pq']:.4f} -> {f['cv_pq']:.4f}")
    return {"post": str(best[1].relative_to(ROOT)), "mode": best[2], "val_post": best[0],
            "filter": str(filt.relative_to(ROOT)), "val_filtered": f["cv_pq"]}


def build_submission(name: str, probs_test: Path, filt: str) -> Path:
    out = ROOT / "outputs" / f"submission_{name}.csv"
    sh([PY, "scripts/make_submission.py", "--probs", probs_test, "--filter", ROOT / filt, "--out", out])
    return out


def submit(csv: Path, message: str) -> str | None:
    """Submit and wait for the public score; None when Kaggle refused (e.g. daily limit)."""
    r = sh(["kaggle", "competitions", "submit", COMPETITION, "-f", csv, "-m", message], check=False)
    if r.returncode != 0 or "Successfully" not in r.stdout:
        log(f"submit refused: {(r.stdout + r.stderr).strip()[-300:]}")
        return None
    for _ in range(40):
        time.sleep(15)
        rows = subprocess.run(["kaggle", "competitions", "submissions", COMPETITION],
                              capture_output=True, text=True).stdout.splitlines()
        row = next((x for x in rows if message[:40] in x), "")
        if row and "pending" not in row.lower():
            return row.split()[-1] if "COMPLETE" in row else "error"
    return "pending"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--owner", default="nguyenchicuong")
    ap.add_argument("--poll", type=int, default=600, help="seconds between polls")
    ap.add_argument("--hours", type=float, default=36)
    args = ap.parse_args()

    st = load_state()
    deadline = time.time() + args.hours * 3600
    while time.time() < deadline:
        for run in args.runs:
            rs = st["runs"].setdefault(run, {})
            if rs.get("errors", 0) >= 3:
                continue  # needs a look; fix, clear its entry in watch_state.json, restart
            try:
                if "val_filtered" not in rs and dataset_ready(args.owner, run):
                    log(f"{run}: dataset ready, processing")
                    out = download(args.owner, run)
                    modes = ["cc", "group"] if (out / "probs_val" / "group").is_dir() else ["cc"]
                    rs.update(tune_and_filter(run, out / "probs_val", modes))
                    rs["csv"] = str(build_submission(run, out / "probs_test", rs["filter"]).relative_to(ROOT))
                    save_state(st)
                if "val_filtered" in rs and "lb" not in rs:
                    lb = submit(ROOT / rs["csv"], f"{run} tta + FP filter (val {rs['val_filtered']:.3f})")
                    if lb is not None:
                        rs["lb"] = lb
                        log(f"{run}: public LB {lb}")
                    save_state(st)
            except Exception as e:  # keep watching the other runs
                log(f"{run}: ERROR {e}")
                rs["error"] = str(e)[-500:]
                rs["errors"] = rs.get("errors", 0) + 1
                save_state(st)

        done = [r for r in args.runs if "val_filtered" in st["runs"].get(r, {})]
        key = "+".join(sorted(done))
        ens = st["ensemble"]
        if len(done) >= 2 and ens.get("key") != key:
            try:
                name = "ens_" + "_".join(r.replace("unet1024_", "") for r in sorted(done))
                maps = {}
                for which in ("val", "test"):
                    out = ROOT / "outputs" / "probs" / f"{name}_{which}"
                    sh([PY, "scripts/ensemble_probs.py", "--inputs",
                        *[ROOT / "outputs" / "colab" / r / f"probs_{which}" for r in sorted(done)], "--out", out])
                    maps[which] = out
                res = tune_and_filter(name, maps["val"], ["cc"])
                best_single = max(st["runs"][r]["val_filtered"] for r in done)
                ens.clear()
                ens.update(key=key, name=name, best_single=best_single, **res)
                if res["val_filtered"] > best_single:
                    ens["csv"] = str(build_submission(name, maps["test"], res["filter"]).relative_to(ROOT))
                else:
                    log(f"{name}: val {res['val_filtered']:.4f} <= best single {best_single:.4f}, not submitted")
                save_state(st)
            except Exception as e:
                log(f"ensemble: ERROR {e}")
                ens.update(key=key, error=str(e)[-500:])
                save_state(st)
        if ens.get("csv") and "lb" not in ens:
            lb = submit(ROOT / ens["csv"], f"{ens['name']} + FP filter (val {ens['val_filtered']:.3f})")
            if lb is not None:
                ens["lb"] = lb
                log(f"{ens['name']}: public LB {lb}")
            save_state(st)

        if all("lb" in st["runs"].get(r, {}) or st["runs"].get(r, {}).get("errors", 0) >= 3 for r in args.runs) and ("csv" not in ens or "lb" in ens):
            log("all runs processed")
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
