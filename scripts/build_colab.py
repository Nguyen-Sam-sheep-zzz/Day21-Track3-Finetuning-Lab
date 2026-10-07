#!/usr/bin/env python3
"""Generate colab/*.ipynb from notebooks/*.py (jupytext py:percent).

The .py files are the source of truth. The Colab notebooks are generated, plus a
bootstrap cell that clones the repo and installs deps — Colab starts with no repo.

Usage: python scripts/build_colab.py    (needs jupytext)
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks"
OUT = ROOT / "colab"

BOOTSTRAP = """# @title Setup (chạy ô này trước)
# Clone fork chứa labkit.evidence; giữ nguyên checkout trong suốt thí nghiệm frozen.
import os, pathlib, subprocess, sys

REPO = "https://github.com/Nguyen-Sam-sheep-zzz/Day21-Track3-Finetuning-Lab.git"
BRANCH = "feature/lab21-finetuning"
HERE = pathlib.Path.cwd()
REPO_DIR = HERE if (HERE / ".git").exists() else HERE / "Day21-Track3-Finetuning-Lab"
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "-q", "--branch", BRANCH, "--single-branch",
                    REPO, str(REPO_DIR)], check=True)
if not (REPO_DIR / ".git").exists():
    raise RuntimeError("Setup cần checkout Git riêng; dùng runtime/thư mục mới trước NB2.")
origin = subprocess.run(["git", "config", "--get", "remote.origin.url"], cwd=REPO_DIR,
                        capture_output=True, text=True, check=True).stdout.strip()
branch = subprocess.run(["git", "branch", "--show-current"], cwd=REPO_DIR,
                        capture_output=True, text=True, check=True).stdout.strip()
if origin.removesuffix(".git") != REPO.removesuffix(".git") or branch != BRANCH:
    raise RuntimeError("Checkout không đúng fork/branch. Chọn checkout đúng trước NB2; không đổi code giữa run frozen.")
if not (REPO_DIR / "src" / "labkit" / "evidence.py").exists():
    raise RuntimeError("Checkout thiếu labkit.evidence. Dùng bản feature/lab21-finetuning đã push trước NB2.")
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR / "src"))
# Setup không pull/fetch/checkout/reset trên checkout đã tồn tại.

# Install from requirements.txt, NOT a copied list. The copied list is how the
# torchao>=0.16 pin reached requirements.txt and this bootstrap on different days --
# and a bootstrap missing a pin does not fail here, it fails 10 minutes later inside
# get_peft_model(). One source of truth. torch is preinstalled on Colab and
# requirements.txt pins it compatibly, so that line is a no-op.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
               check=True)

os.environ.setdefault("COMPUTE_TIER", "T4")
import torch
print("commit:", subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — Runtime > Change runtime type > T4 GPU")
"""


def _stamp_cell_ids(raw: dict) -> None:
    """Give every cell an id derived from its position and content.

    nbformat mints a RANDOM id per cell, so regenerating unchanged notebooks produced a
    ~90-line diff of nothing but id churn. That is not cosmetic: a diff that is always
    noise is a diff nobody reads, which is how the stale-bootstrap bug (F-18) survived
    a review. Now `make colab` on unchanged sources is a genuinely empty diff, and any
    line that does move is a line that means something.
    """
    for i, cell in enumerate(raw.get("cells", [])):
        body = "".join(cell.get("source", []))
        cell["id"] = hashlib.sha1(f"{i}\x00{body}".encode()).hexdigest()[:8]


def main() -> int:
    try:
        import jupytext
    except ImportError:
        print("jupytext not installed:  pip install jupytext", file=sys.stderr)
        return 1

    OUT.mkdir(exist_ok=True)
    made = []
    for src in sorted(SRC.glob("*.py")):
        nb = jupytext.read(src, fmt="py:percent")
        import nbformat
        nb.cells.insert(0, nbformat.v4.new_code_cell(BOOTSTRAP))
        dest = OUT / f"Lab21_{src.stem}.ipynb"
        # ensure_ascii=True: Vietnamese content must survive tooling that assumes ASCII
        jupytext.write(nb, dest, fmt="ipynb")
        raw = json.loads(dest.read_text(encoding="utf-8"))
        _stamp_cell_ids(raw)
        dest.write_text(json.dumps(raw, ensure_ascii=True, indent=1), encoding="utf-8")
        made.append(dest.name)
    print(f"wrote {len(made)} notebooks to colab/:")
    for m in made:
        print("  ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
