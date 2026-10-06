#!/usr/bin/env python3
"""
examples/custom_calibration.py
Calibrate a GeoScorer on your own task from 50 labeled examples.

This script shows the full calibration workflow:
  1. Load your model
  2. Provide 50 (question, correct_answer, wrong_answers) triples
  3. Calibrate the direction
  4. Save and reload
  5. Score new examples

For illustration, this uses a subset of ARC-Challenge as the "custom" task,
but the same workflow applies to any domain-specific Q&A dataset.

Usage:
    python examples/custom_calibration.py
    python examples/custom_calibration.py --model Qwen/Qwen2.5-7B-Instruct --calib_n 50
"""

import argparse
import numpy as np
import tempfile
from datasets import load_dataset
from tqdm import tqdm

from geoscore import GeoScorer


def load_arc_examples(calib_n, test_n, seed):
    rng      = np.random.default_rng(seed)
    train_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    test_ds  = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    ci = rng.choice(len(train_ds), size=min(calib_n, len(train_ds)), replace=False)
    ti = rng.choice(len(test_ds),  size=min(test_n,  len(test_ds)),  replace=False)

    def to_calib(ex):
        labels = ex["choices"]["label"]
        texts  = ex["choices"]["text"]
        cidx   = labels.index(ex["answerKey"])
        return {"question": ex["question"], "correct": texts[cidx],
                "wrongs": [t for j, t in enumerate(texts) if j != cidx]}

    def to_test(ex):
        labels = ex["choices"]["label"]
        texts  = ex["choices"]["text"]
        cidx   = labels.index(ex["answerKey"])
        return {"question": ex["question"], "choices": texts, "answer_idx": cidx}

    return ([to_calib(train_ds[int(i)]) for i in ci],
            [to_test(test_ds[int(i)])   for i in ti])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--calib_n",  type=int, default=50)
    ap.add_argument("--test_n",   type=int, default=200)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    # ── Step 1: Load model ────────────────────────────────────────────────────
    print(f"\nStep 1: Loading {args.model}...")
    scorer = GeoScorer(args.model)
    print(scorer)

    # ── Step 2: Load calibration examples ────────────────────────────────────
    print(f"\nStep 2: Loading {args.calib_n} calibration examples...")
    calib, test = load_arc_examples(args.calib_n, args.test_n, args.seed)
    print(f"  Calibration: {len(calib)}  |  Test: {len(test)}")

    # ── Step 3: Calibrate ─────────────────────────────────────────────────────
    print(f"\nStep 3: Calibrating direction from {len(calib)} examples...")
    scorer.calibrate(calib)
    print(scorer)

    # ── Step 4: Save and reload ───────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    print(f"\nStep 4: Saving to {path} and reloading...")
    scorer.save(path)

    reloaded = GeoScorer.load(args.model, path)
    print(f"  Reloaded: {reloaded}")

    # ── Step 5: Evaluate on test examples ─────────────────────────────────────
    print(f"\nStep 5: Evaluating on {len(test)} test examples...")
    hits = 0
    for ex in tqdm(test, desc="  scoring"):
        ranked = reloaded.rank(ex["question"], ex["choices"])
        if ranked[0][0] == ex["choices"][ex["answer_idx"]]:
            hits += 1

    acc = hits / len(test) * 100
    print(f"\n  Accuracy: {acc:.1f}%   (zero-shot baseline ~41% for this model)")
    print(f"\nFull workflow complete. The direction file at {path} can now be")
    print(f"deployed with GeoScorer.load('{args.model}', '{path}').\n")


if __name__ == "__main__":
    main()
