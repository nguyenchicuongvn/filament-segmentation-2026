"""Bundle the code the Colab notebook needs into build/colab/filseg-colab.zip.

Upload the zip to MyDrive/kaggle/filament-segmentation-2026/ whenever the code changes.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "colab" / "filseg-colab.zip"
INCLUDE = ["src", "scripts", "configs", "requirements.txt"]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in INCLUDE:
            p = ROOT / name
            files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
            for f in files:
                if "__pycache__" in f.parts:
                    continue
                z.write(f, f.relative_to(ROOT))
                n += 1
    print(f"{n} files -> {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
