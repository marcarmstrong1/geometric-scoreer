"""
geoscore/core.py
Core computation: hidden state extraction, contrastive direction,
candidate scoring. All functions are stateless and device-agnostic.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ── Optimal layer fractions from paper (Table A5) ────────────────────────────

OPTIMAL_LAYER_FRACS = {
    "meta-llama/Llama-3.2-1B-Instruct":   0.60,
    "Qwen/Qwen2.5-1.5B-Instruct":         0.65,
    "google/gemma-2-2b-it":               0.70,
    "Qwen/Qwen2.5-7B-Instruct":           0.70,
    "meta-llama/Llama-3.1-8B-Instruct":   0.65,
}

DEFAULT_LAYER_FRAC = 0.70   # safe fallback for unknown models


def resolve_layer(model, frac: float) -> int:
    """Convert a fractional depth to an absolute layer index."""
    return max(1, int(frac * model.config.num_hidden_layers))


def make_prompt(question: str, answer: str) -> str:
    """Standard prompt format used throughout calibration and inference."""
    return f"Question: {question}\nAnswer: {answer}"


@torch.no_grad()
def get_hidden(
    model,
    tokenizer,
    text: str,
    layer_idx: int,
    device: str,
) -> torch.Tensor:
    """
    Single forward pass. Returns the L2-normalised hidden state of the final
    token at the specified layer.

    Args:
        model:      A frozen HuggingFace causal LM.
        tokenizer:  Corresponding tokenizer.
        text:       Input string (formatted prompt).
        layer_idx:  Absolute transformer layer index (1-indexed, 0 = embedding).
        device:     Target device string.

    Returns:
        Tensor of shape (hidden_dim,), float32, L2-normalised.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    out = model(**inputs, output_hidden_states=True)
    h   = out.hidden_states[layer_idx][0, -1].float()
    return h / (h.norm() + 1e-8)


def compute_delta(
    model,
    tokenizer,
    examples: list[dict],
    layer_idx: int,
    device: str,
    verbose: bool = True,
) -> tuple[torch.Tensor, float]:
    """
    Compute the contrastive correctness direction δ̂ from calibration examples.

    Each example must be a dict with keys:
        "question" (str)  — the question text
        "correct"  (str)  — the correct answer text
        "wrongs"   (list) — list of incorrect answer strings

    δ̂ = L2_norm( mean_i( h(Q_i, A_correct_i) − mean_j( h(Q_i, A_wrong_ij) ) ) )

    Args:
        model:      Frozen HuggingFace causal LM.
        tokenizer:  Corresponding tokenizer.
        examples:   List of calibration dicts (see above).
        layer_idx:  Absolute layer index for hidden state extraction.
        device:     Target device.
        verbose:    Show tqdm progress bar.

    Returns:
        (delta_hat, snr): L2-normalised direction tensor and signal-to-noise ratio.
        SNR = ||mean_delta|| / mean_per_dim_std. Values >5× indicate a stable
        direction; the paper reports 8–16× for the tested models.
    """
    deltas = []
    iterator = tqdm(examples, desc="  calibrating", leave=False) if verbose else examples

    for ex in iterator:
        q       = ex["question"]
        correct = ex["correct"]
        wrongs  = ex["wrongs"]

        h_correct   = get_hidden(model, tokenizer, make_prompt(q, correct),  layer_idx, device)
        h_wrong_sum = torch.zeros_like(h_correct)

        for w in wrongs:
            h_wrong_sum += get_hidden(model, tokenizer, make_prompt(q, w), layer_idx, device)

        h_wrong_mean = h_wrong_sum / len(wrongs)
        deltas.append(h_correct - h_wrong_mean)

    stacked    = torch.stack(deltas)              # (N, hidden_dim)
    mean_delta = stacked.mean(dim=0)
    snr        = (mean_delta.norm() / stacked.std(dim=0).mean()).item()
    delta_hat  = mean_delta / (mean_delta.norm() + 1e-8)

    return delta_hat, snr


def score_candidates(
    model,
    tokenizer,
    question: str,
    candidates: list[str],
    delta_hat: torch.Tensor,
    layer_idx: int,
    device: str,
) -> list[tuple[str, float]]:
    """
    Score each candidate answer by its projection onto δ̂.

    Returns a list of (answer, score) tuples sorted by score descending.
    Higher score = stronger alignment with the correctness direction.

    Args:
        model:      Frozen HuggingFace causal LM.
        tokenizer:  Corresponding tokenizer.
        question:   The question string.
        candidates: List of candidate answer strings.
        delta_hat:  L2-normalised correctness direction (from compute_delta).
        layer_idx:  Absolute layer index.
        device:     Target device.

    Returns:
        List of (answer, score) tuples sorted by score descending.
    """
    scored = []
    for answer in candidates:
        h = get_hidden(model, tokenizer, make_prompt(question, answer), layer_idx, device)
        s = torch.dot(h, delta_hat.to(device)).item()
        scored.append((answer, s))

    return sorted(scored, key=lambda x: x[1], reverse=True)


@torch.no_grad()
def confidence_score(
    model,
    tokenizer,
    question: str,
    answer: str,
    delta_hat: torch.Tensor,
    layer_idx: int,
    device: str,
) -> float:
    """
    Return a single scalar confidence score for a (question, answer) pair.

    This is the dot product of the answer's hidden-state representation with
    the correctness direction δ̂. Higher values indicate greater alignment with
    the correctness geometry learned from calibration examples.

    Useful as a lightweight hallucination detection signal: score a generated
    response and flag it for review if the score falls below a threshold.
    The paper reports AUROC 0.693 vs 0.578 for log-probability scoring on
    ARC-Challenge using this signal.

    Args:
        model:     Frozen HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        question:  The question string.
        answer:    The answer string to score.
        delta_hat: L2-normalised correctness direction.
        layer_idx: Absolute layer index.
        device:    Target device.

    Returns:
        Float in approximately [-0.2, 0.2]. Not a calibrated probability.
    """
    h = get_hidden(model, tokenizer, make_prompt(question, answer), layer_idx, device)
    return torch.dot(h, delta_hat.to(device)).item()


@torch.no_grad()
def zeroshot_logprob(
    model,
    tokenizer,
    question: str,
    answer: str,
    device: str,
) -> float:
    """
    Length-normalised log-probability of the answer tokens given the question.
    Used as the zero-shot baseline in the paper.
    """
    prefix  = make_prompt(question, "")
    full    = make_prompt(question, answer)
    n_pfx   = tokenizer(prefix, return_tensors="pt").input_ids.shape[1]
    fids    = tokenizer(full, return_tensors="pt", truncation=True,
                        max_length=512).input_ids.to(device)
    lp      = F.log_softmax(model(fids).logits[0], dim=-1)
    toks    = fids[0, n_pfx:]
    slp     = lp[n_pfx - 1 : n_pfx - 1 + len(toks)]
    if len(toks) == 0:
        return float("-inf")
    return sum(slp[i, toks[i]].item() for i in range(len(toks))) / len(toks)
