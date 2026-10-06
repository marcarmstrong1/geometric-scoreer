"""
geoscore/scorer.py
Main GeoScorer class — the primary public interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import (
    DEFAULT_LAYER_FRAC,
    OPTIMAL_LAYER_FRACS,
    compute_delta,
    confidence_score,
    resolve_layer,
    score_candidates,
)

# HuggingFace Hub repo where pre-computed directions are stored.
# Update this to your actual HF username after pushing the direction files.
HF_DIRECTIONS_REPO = "YOUR_HF_USERNAME/geoscore-directions"

# Local cache: directions/ folder inside the installed package
_DIRECTIONS_DIR = Path(__file__).parent.parent / "directions"

# Canonical shorthands used as folder names in the directions repo
_MODEL_SHORTHANDS = {
    "meta-llama/Llama-3.2-1B-Instruct":  "llama-1b",
    "Qwen/Qwen2.5-1.5B-Instruct":        "qwen-1.5b",
    "google/gemma-2-2b-it":              "gemma-2b",
    "Qwen/Qwen2.5-7B-Instruct":          "qwen-7b",
    "meta-llama/Llama-3.1-8B-Instruct":  "llama-8b",
}


def _get_direction_path(model_name: str, task: str) -> Path:
    """
    Return path to a direction file, downloading from HuggingFace Hub if
    it is not already cached locally.

    Download order:
      1. Local directions/ folder (inside the package — present after cloning
         and running precompute_directions.py, or if committed to the repo)
      2. HuggingFace Hub cache (~/.cache/huggingface/hub/...)
    """
    shorthand = _MODEL_SHORTHANDS.get(model_name)
    if shorthand is None:
        raise ValueError(
            f"No pre-computed direction for '{model_name}'.\n"
            f"Supported models: {list(_MODEL_SHORTHANDS.keys())}\n"
            f"To calibrate from scratch: GeoScorer(model).calibrate(examples)"
        )

    filename  = f"{shorthand}/{task}.pt"
    local     = _DIRECTIONS_DIR / shorthand / f"{task}.pt"

    # Fast path: file already present locally (committed or pre-generated)
    if local.exists():
        return local

    # Download from HuggingFace Hub and return the cached path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download pre-computed directions.\n"
            "Install it with: pip install huggingface-hub\n"
            f"Or clone the repo and run: python scripts/precompute_directions.py "
            f"--model {model_name} --task {task}"
        )

    print(f"  Downloading direction for {shorthand}/{task} from HuggingFace Hub...")
    cached = hf_hub_download(
        repo_id   = HF_DIRECTIONS_REPO,
        filename  = filename,
        repo_type = "model",
    )
    return Path(cached)


class GeoScorer:
    """
    Geometric answer scorer using a contrastive correctness direction.

    The scorer wraps a frozen language model and a direction vector δ̂.
    Once calibrated (or loaded from a pre-computed direction), it can:
      - rank candidate answers by geometric score
      - return a scalar confidence signal for hallucination detection

    Typical usage:

        # Pre-computed direction — downloaded automatically on first call
        scorer = GeoScorer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", task="arc")
        ranked = scorer.rank(question, choices)

        # Custom calibration — 50 examples, ~5 minutes on a consumer GPU
        scorer = GeoScorer("Qwen/Qwen2.5-1.5B-Instruct")
        scorer.calibrate(examples)  # list of {question, correct, wrongs}
        scorer.save("my_task.pt")
    """

    def __init__(
        self,
        model_name: str,
        layer_frac: Optional[float] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Load a model without a direction. Call .calibrate() or use
        .from_pretrained() / .load() to attach a direction before scoring.

        Args:
            model_name:  HuggingFace model ID.
            layer_frac:  Fractional depth for hidden state extraction (0–1).
                         Defaults to the paper-validated optimum for the model,
                         or 0.70 for unknown models.
            device:      "cuda", "cpu", or None (auto-detect).
            dtype:       Model dtype. bfloat16 recommended for GPU.
        """
        self.model_name = model_name
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_frac = layer_frac or OPTIMAL_LAYER_FRACS.get(model_name, DEFAULT_LAYER_FRAC)

        self._delta_hat: Optional[torch.Tensor] = None
        self._snr:       Optional[float]        = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        self.layer_idx = resolve_layer(self.model, self.layer_frac)

    # ── Class-method constructors ─────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, model_name: str, task: str, **kwargs) -> "GeoScorer":
        """
        Load a scorer with a pre-computed correctness direction.

        On first call for a given (model, task) pair, the direction file is
        downloaded from HuggingFace Hub (~6–16 kB) and cached locally.
        Subsequent calls load from cache instantly.

        Pre-computed directions are available for the five models in the paper
        on tasks 'arc' and 'truthfulqa'.

        Args:
            model_name:  HuggingFace model ID.
            task:        "arc" or "truthfulqa".
            **kwargs:    Passed to GeoScorer.__init__ (device, dtype, etc.).

        Returns:
            A GeoScorer ready to call .rank() and .confidence().
        """
        scorer = cls(model_name, **kwargs)
        path   = _get_direction_path(model_name, task)
        scorer._load_delta(path)
        return scorer

    @classmethod
    def load(cls, model_name: str, path: str, **kwargs) -> "GeoScorer":
        """
        Load a scorer with a direction saved from a previous .calibrate() call.

        Args:
            model_name:  HuggingFace model ID.
            path:        Path to a .pt file produced by .save().
            **kwargs:    Passed to GeoScorer.__init__.
        """
        scorer = cls(model_name, **kwargs)
        scorer._load_delta(path)
        return scorer

    # ── Core methods ─────────────────────────────────────────────────────────

    def calibrate(
        self,
        examples: list[dict],
        verbose: bool = True,
    ) -> "GeoScorer":
        """
        Compute the contrastive correctness direction from calibration examples.

        Each example must be a dict with:
            "question" (str)   — the question text
            "correct"  (str)   — the correct answer text
            "wrongs"   (list)  — list of incorrect answer strings (≥1)

        50 examples is sufficient for a stable direction (SNR typically 8–16×).

        Args:
            examples:  List of calibration dicts.
            verbose:   Print progress bar and SNR.

        Returns:
            self (for chaining).
        """
        if len(examples) == 0:
            raise ValueError("examples cannot be empty.")

        self._delta_hat, self._snr = compute_delta(
            self.model,
            self.tokenizer,
            examples,
            self.layer_idx,
            self.device,
            verbose=verbose,
        )

        if verbose:
            print(f"  δ̂ computed from {len(examples)} examples  |  SNR = {self._snr:.1f}×")

        return self

    def rank(
        self,
        question: str,
        candidates: list[str],
    ) -> list[tuple[str, float]]:
        """
        Score and rank a list of candidate answers.

        Returns a list of (answer, score) tuples sorted by score descending.
        The top-ranked answer is the one most aligned with the correctness
        direction. Use .predict() if you only need the best answer.

        Args:
            question:   The question string.
            candidates: List of candidate answer strings.

        Returns:
            List of (answer, score) sorted descending by score.
        """
        self._require_delta()
        return score_candidates(
            self.model,
            self.tokenizer,
            question,
            candidates,
            self._delta_hat,
            self.layer_idx,
            self.device,
        )

    def predict(self, question: str, candidates: list[str]) -> str:
        """Return the single best candidate answer."""
        return self.rank(question, candidates)[0][0]

    def confidence(self, question: str, answer: str) -> float:
        """
        Return a scalar confidence score for a single (question, answer) pair.

        Higher values indicate stronger alignment with the correctness direction.
        Useful as a hallucination detection signal in production pipelines.

        The paper reports AUROC 0.693 vs 0.578 for log-probability scoring on
        ARC-Challenge using this signal (Qwen-2.5-1.5B, ARC-calibrated direction).

        Returns:
            Float in approximately [-0.2, 0.2]. Not a calibrated probability.
        """
        self._require_delta()
        return confidence_score(
            self.model,
            self.tokenizer,
            question,
            answer,
            self._delta_hat,
            self.layer_idx,
            self.device,
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save the calibrated direction to a .pt file for later reuse.

        The saved file contains the direction vector and metadata. Reload with
        GeoScorer.load(model_name, path).

        Args:
            path:  Output file path (should end in .pt).
        """
        self._require_delta()
        torch.save(
            {
                "delta_hat":  self._delta_hat.cpu(),
                "snr":        self._snr,
                "layer_frac": self.layer_frac,
                "layer_idx":  self.layer_idx,
                "model_name": self.model_name,
            },
            path,
        )
        size_kb = Path(path).stat().st_size / 1024
        print(f"  Saved to {path}  ({size_kb:.1f} kB, SNR={self._snr:.1f}×)")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def snr(self) -> Optional[float]:
        """Signal-to-noise ratio of the calibrated direction."""
        return self._snr

    @property
    def direction(self) -> Optional[torch.Tensor]:
        """The L2-normalised correctness direction vector δ̂."""
        return self._delta_hat

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_delta(self, path) -> None:
        data            = torch.load(path, map_location="cpu", weights_only=True)
        self._delta_hat = data["delta_hat"].float()
        self._snr       = data.get("snr")
        if "layer_frac" in data and self.layer_frac == OPTIMAL_LAYER_FRACS.get(
            self.model_name, DEFAULT_LAYER_FRAC
        ):
            self.layer_frac = data["layer_frac"]
            self.layer_idx  = data["layer_idx"]

    def _require_delta(self) -> None:
        if self._delta_hat is None:
            raise RuntimeError(
                "No correctness direction loaded.\n"
                "Options:\n"
                "  1. GeoScorer.from_pretrained(model, task='arc')  — auto-download\n"
                "  2. scorer.calibrate(examples)                    — from your data\n"
                "  3. GeoScorer.load(model, path)                   — from saved file"
            )

    def __repr__(self) -> str:
        has_dir = self._delta_hat is not None
        snr_str = f", SNR={self._snr:.1f}×" if self._snr else ""
        return (
            f"GeoScorer(model={self.model_name!r}, "
            f"layer={self.layer_frac} → abs {self.layer_idx}, "
            f"direction={'loaded' + snr_str if has_dir else 'not loaded'})"
        )
