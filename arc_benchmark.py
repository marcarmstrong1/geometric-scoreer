#!/usr/bin/env python3
"""
examples/arc_benchmark.py
Reproduce paper ARC-Challenge results for a given model.

Compares GeoScorer against zero-shot log-probability on the same
held-out test examples, using a pre-computed direction.

Usage:
    python examples/arc_benchmark.py
    python examples/arc_benchmark.py --model Qwen/Qwen2.5-7B-Instruct
    python examples/arc_benchmark.py --model meta-llama/Llama-3.1-8B-Instruct --test_n 500
"""

import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from geoscore import GeoScorer, zeroshot_logprob

PAPER_RESULTS = {
    "meta-llama/Llama-3.2-1B-Instruct":  {"zs": 35.0, "grm": 44.7},
    "Qwen/Qwen2.5-1.5B-Instruct":        {"zs": 41.0, "grm": 61.1},
    "google/gemma-2-2b-it":              {"zs": 44.4, "grm": 66.5},
    "Qwen/Qwen2.5-7B-Instruct":          {"zs": 48.0, "grm": 80.0},
    "meta-llama/Llama-3.1-8B-Instruct":  {"zs": 45.6, "grm": 59.0},
}


def load_arc_test(n, seed):
    ds  = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    out = []
    for i in idx:
        ex     = ds[int(i)]
        labels = ex["choices"]["label"]
        texts  = ex["choices"]["text"]
        cidx   = labels.index(ex["answerKey"])
        out.append({"question": ex["question"], "choices": texts, "answer_idx": cidx})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--test_n", type=int, default=500)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    print(f"\nLoading {args.model}...")
    scorer = GeoScorer.from_pretrained(args.model, task="arc")
    print(scorer)

    print(f"\nLoading ARC-Challenge test ({args.test_n} examples)...")
    test = load_arc_test(args.test_n, args.seed)

    grm_hits = zs_hits = 0

    for ex in tqdm(test, desc="Evaluating"):
        q       = ex["question"]
        choices = ex["choices"]
        cidx    = ex["answer_idx"]

        # GRM scoring
        ranked = scorer.rank(q, choices)
        if ranked[0][0] == choices[cidx]:
            grm_hits += 1

        # Zero-shot baseline
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
    print(f"  ARC-Challenge  |  {args.model}")
    print(f"{'='*W}")
    print(f"  Zero-shot      : {zs_acc:5.1f}%")
    print(f"  GeoScore (GRM) : {grm_acc:5.1f}%   (+{grm_acc - zs_acc:.1f} pp)")
    print(f"  Random baseline:  25.0%")

    if args.model in PAPER_RESULTS:
        p = PAPER_RESULTS[args.model]
        print(f"\n  Paper reports  : ZS={p['zs']:.1f}%  GRM={p['grm']:.1f}%")
        print(f"  Deviation      : ZS Δ={zs_acc - p['zs']:+.1f}pp  "
              f"GRM Δ={grm_acc - p['grm']:+.1f}pp")
        print(f"  (small deviations expected from test-set sampling)")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
