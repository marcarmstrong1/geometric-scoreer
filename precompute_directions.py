#!/usr/bin/env python3
"""
scripts/precompute_directions.py
Generate pre-computed direction files for a model and task.

This script produces the .pt files that GeoScorer.from_pretrained() loads.
Run this once per (model, task) combination and commit the output to the
directions/ folder (files are tiny — <20kB each).

Usage:
    # Generate ARC direction for Qwen-1.5B
    python scripts/precompute_directions.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --task arc

    # Generate all directions for all paper models
    python scripts/precompute_directions.py --all
"""

import argparse
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset

from geoscore import GeoScorer
from geoscore.scorer import _DIRECTIONS_DIR, _MODEL_SHORTHANDS

PAPER_MODELS = list(_MODEL_SHORTHANDS.keys())
TASKS        = ["arc", "truthfulqa"]
CALIB_N      = 50
SEED         = 42


def load_calib_arc(seed):
    ds  = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(len(ds), size=min(CALIB_N, len(ds)), replace=False)
    out = []
    for i in idx:
        ex     = ds[int(i)]
        labels = ex["choices"]["label"]
        texts  = ex["choices"]["text"]
        cidx   = labels.index(ex["answerKey"])
        out.append({"question": ex["question"], "correct": texts[cidx],
                    "wrongs": [t for j, t in enumerate(texts) if j != cidx]})
    return out


def load_calib_truthfulqa(seed):
    ds  = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:CALIB_N]
    out = []
    for i in idx:
        ex      = ds[int(i)]
        choices = ex["mc1_targets"]["choices"]
        labels  = ex["mc1_targets"]["labels"]
        cidx    = labels.index(1)
        out.append({"question": ex["question"], "correct": choices[cidx],
                    "wrongs": [c for j, c in enumerate(choices) if j != cidx]})
    return out


LOADERS = {"arc": load_calib_arc, "truthfulqa": load_calib_truthfulqa}


def generate(model_name, task):
    shorthand = _MODEL_SHORTHANDS.get(model_name)
    if shorthand is None:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Known: {list(_MODEL_SHORTHANDS.keys())}")

    out_dir  = _DIRECTIONS_DIR / shorthand
    out_path = out_dir / f"{task}.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"  Already exists: {out_path}  (delete to regenerate)")
        return

    print(f"\n{'─'*60}")
    print(f"  Model : {model_name}")
    print(f"  Task  : {task}")
    print(f"  Output: {out_path}")
    print(f"{'─'*60}")

    calib  = LOADERS[task](SEED)
    scorer = GeoScorer(model_name)
    scorer.calibrate(calib, verbose=True)
    scorer.save(str(out_path))
    print(f"  ✓ Saved  ({out_path.stat().st_size / 1024:.1f} kB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="HuggingFace model ID (or use --all)")
    ap.add_argument("--task",  default="arc", choices=TASKS)
    ap.add_argument("--all",   action="store_true",
                    help="Generate all directions for all paper models and tasks")
    args = ap.parse_args()

    if args.all:
        for model in PAPER_MODELS:
            for task in TASKS:
                generate(model, task)
        print("\nAll directions generated.")
    else:
        if args.model is None:
            ap.error("Provide --model or use --all")
        generate(args.model, args.task)


if __name__ == "__main__":
    main()
