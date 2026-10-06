#!/usr/bin/env python3
"""
examples/hallucination_detection.py
Demonstrate GeoScorer as a hallucination detector.

Treats every (question, answer) pair in ARC-Challenge as a binary
classification instance: is this answer correct or not?
Reports AUROC for GeoScorer vs zero-shot log-probability.

Paper result: GeoScorer 0.693 AUROC vs 0.578 log-prob (Qwen-2.5-1.5B).

Usage:
    python examples/hallucination_detection.py
    python examples/hallucination_detection.py --model google/gemma-2-2b-it
"""

import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from geoscore import GeoScorer, zeroshot_logprob


def compute_auroc(scored_pairs):
    """AUROC = P(score_positive > score_negative) over all pos/neg pairs."""
    pos = sorted([s for s, l in scored_pairs if l == 1], reverse=True)
    neg = sorted([s for s, l in scored_pairs if l == 0], reverse=True)
    if not pos or not neg:
        return 0.5
    concordant = 0
    neg_ptr    = 0
    for p in pos:
        while neg_ptr < len(neg) and neg[neg_ptr] >= p:
            neg_ptr += 1
        concordant += len(neg) - neg_ptr
    return concordant / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n",      type=int, default=500)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    print(f"\nLoading {args.model}...")
    scorer = GeoScorer.from_pretrained(args.model, task="arc")
    print(scorer)

    ds  = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)

    grm_pairs = []
    zs_pairs  = []

    for i in tqdm(idx, desc="Scoring"):
        ex     = ds[int(i)]
        labels = ex["choices"]["label"]
        texts  = ex["choices"]["text"]
        q      = ex["question"]
        cidx   = labels.index(ex["answerKey"])

        for j, choice in enumerate(texts):
            label = 1 if j == cidx else 0
            grm_pairs.append((scorer.confidence(q, choice), label))
            zs_pairs.append((
                zeroshot_logprob(scorer.model, scorer.tokenizer, q, choice, scorer.device),
                label,
            ))

    grm_auroc = compute_auroc(grm_pairs)
    zs_auroc  = compute_auroc(zs_pairs)

    grm_pos = np.mean([s for s, l in grm_pairs if l == 1])
    grm_neg = np.mean([s for s, l in grm_pairs if l == 0])

    W = 60
    print(f"\n{'='*W}")
    print(f"  Hallucination Detection  |  {args.model}")
    print(f"  {len(grm_pairs)} (question, answer) pairs  "
          f"({sum(l for _, l in grm_pairs)} correct / "
          f"{sum(1-l for _, l in grm_pairs)} wrong)")
    print(f"{'='*W}")
    print(f"  Random baseline    : 0.500")
    print(f"  Zero-shot log-prob : {zs_auroc:.3f}   (+{(zs_auroc-0.5)*100:.1f} pp)")
    print(f"  GeoScore           : {grm_auroc:.3f}   (+{(grm_auroc-0.5)*100:.1f} pp)")
    print(f"\n  Score means — correct: {grm_pos:+.4f}  wrong: {grm_neg:+.4f}  "
          f"separation: {grm_pos-grm_neg:+.4f}")
    print(f"{'='*W}")
    print(f"\n  Paper reports: GeoScore 0.693  log-prob 0.578  (Qwen-2.5-1.5B)\n")


if __name__ == "__main__":
    main()
