"""
geoscore — Geometric answer scoring for language models.

Score answers and flag hallucinations using a contrastive correctness
direction extracted from 50 calibration examples. No training required.

Quick start:

    from geoscore import GeoScorer

    # Pre-computed direction
    scorer = GeoScorer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", task="arc")
    ranked = scorer.rank(question, choices)
    conf   = scorer.confidence(question, answer)

    # Custom calibration
    scorer = GeoScorer("your-model")
    scorer.calibrate(examples)   # list of {question, correct, wrongs}
    scorer.save("my_task.pt")

See https://github.com/YOUR_USERNAME/geoscore for full documentation.
"""

from .scorer import GeoScorer
from .core import (
    compute_delta,
    score_candidates,
    confidence_score,
    zeroshot_logprob,
    get_hidden,
    make_prompt,
    resolve_layer,
    OPTIMAL_LAYER_FRACS,
)

__version__ = "0.1.0"
__all__     = [
    "GeoScorer",
    "compute_delta",
    "score_candidates",
    "confidence_score",
    "zeroshot_logprob",
    "get_hidden",
    "make_prompt",
    "resolve_layer",
    "OPTIMAL_LAYER_FRACS",
]
