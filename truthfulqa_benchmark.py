#!/usr/bin/env python3
"""
examples/truthfulqa_benchmark.py
Reproduce paper TruthfulQA results for a given model.

Uses the pre-computed TruthfulQA-specific direction. Zero-shot baseline
on the same examples shows the magnitude of the improvement.

Usage:
    python examples/truthfulqa_benchmark.py
    python examples/truthfulqa_benchmark.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from geoscore import GeoScorer, zeroshot_logprob

PAPER_RESULTS = {
    "meta-llama/Llama-3.2-1B-Instruct":  {"zs": 32.5, "grm": 75.9},
    "Qwen/Qwen2.5-1.5B-Instruct":        {"zs": 30.6, "grm": 72.5},
    "google/gemma-2-2b-it":              {"zs": 38.5, "grm": 76.5},
    "Qwen/Qwen2.5-7B-Instruct":          {"zs": 39.5, "grm": 81.2},
    "meta-llama/Llama-3.1-8B-Instruct":  {"zs": 28.7, "grm": 80.4},
}

CALIB_N = 50   # examples used to build the pre-computed direction


def load_truthfulqa_test(calib_n, seed):
    """Load test split — first calib_n examples excluded (used for direction)."""
    ds  = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds)).tolist()

    def to_test(i):
        ex      = ds[int(i)]
        choices = ex["mc1_targets"]["choices"]
        labels  = ex["mc1_targets"]["labels"]
        cidx    = labels.index(1)
        return {"question": ex["question"], "choices": choices, "answer_idx": cidx}

    return [to_test(i) for i in perm[calib_n:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()

    print(f"\nLoading {args.model}...")
    scorer = GeoScorer.from_pretrained(args.model, task="truthfulqa")
    print(scorer)

    test = load_truthfulqa_test(CALIB_N, args.seed)
    print(f"TruthfulQA test: {len(test)} examples (first {CALIB_N} held out for direction)")

    grm_hits = zs_hits = 0

    for ex in tqdm(test, desc="Evaluating"):
        q       = ex["question"]
        choices = ex["choices"]
        cidx    = ex["answer_idx"]

        ranked = scorer.rank(q, choices)
        if ranked[0][0] == choices[cidx]:
            grm_hits += 1

        zs_scores = [
            zeroshot_logprob(scorer.model, scorer.tokenizer, q, c, scorer.device)
            for c in choices
        ]
        if int(np.argmax(zs_scores)) == cidx:
            zs_hits += 1

    n       = len(test)
    grm_acc = grm_hits / n * 100
    zs_acc  = zs_hits  / n * 100

    W = 60
    print(f"\n{'='*W}")
    print(f"  TruthfulQA  |  {args.model}")
    print(f"{'='*W}")
    print(f"  Zero-shot      : {zs_acc:5.1f}%   (near-chance; log-prob rewards confidence)")
    print(f"  GeoScore (GRM) : {grm_acc:5.1f}%   (+{grm_acc - zs_acc:.1f} pp)")

    if args.model in PAPER_RESULTS:
        p = PAPER_RESULTS[args.model]
        print(f"\n  Paper reports  : ZS={p['zs']:.1f}%  GRM={p['grm']:.1f}%")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
