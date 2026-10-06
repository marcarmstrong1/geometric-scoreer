# geoscore

**Score answers. Flag hallucinations. No training required.**

`geoscore` extracts a *correctness direction* from a frozen language model
using 50 labeled examples, then scores candidate answers with a single dot
product at inference time. No generation. No fine-tuning. No secondary model.

```
pip install geoscore
```

---

## Results

Improvement over zero-shot log-probability scoring on the same held-out test
examples, using pre-computed directions (50 calibration examples, no training):

| Model | ARC-Challenge | TruthfulQA |
|-------|:---:|:---:|
| Llama-3.2-1B-Instruct | +9.7 pp | +43.4 pp |
| Qwen-2.5-1.5B-Instruct | +20.1 pp | +41.9 pp |
| Gemma-2-2B-IT | +22.1 pp | +38.0 pp |
| Qwen-2.5-7B-Instruct | +32.0 pp | +41.7 pp |
| Llama-3.1-8B-Instruct | +13.4 pp | **+51.7 pp** |

A logistic regression probe trained on the identical 50 calibration examples
matches `geoscore` within 2.1 pp across all configurations — confirming the
mean direction captures the full discriminative signal without optimisation.

---

## Quick start

### Rank answer candidates (multiple-choice, QA, retrieval)

```python
from geoscore import GeoScorer

# Load a pre-computed direction — zero setup, works immediately
scorer = GeoScorer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", task="arc")

question = "Which organelle is responsible for producing energy in a cell?"
choices  = ["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"]

ranked = scorer.rank(question, choices)
# ranked = [("Mitochondria", 0.041), ("Nucleus", -0.003), ...]

print(ranked[0][0])   # "Mitochondria"
```

### Flag potential hallucinations (single-answer deployment)

```python
from geoscore import GeoScorer

scorer = GeoScorer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", task="arc")

question      = "What is the powerhouse of the cell?"
good_answer   = "The mitochondria produce ATP through cellular respiration."
bad_answer    = "The nucleus controls all energy production in the cell."

print(scorer.confidence(question, good_answer))   #  0.038  (higher = more confident)
print(scorer.confidence(question, bad_answer))    # -0.012  (lower = less confident)

# In production: flag responses below a threshold for human review
THRESHOLD = 0.0
response = model.generate(question)
if scorer.confidence(question, response) < THRESHOLD:
    flag_for_review(response)
```

### Calibrate on your own task (50 examples, ~5 minutes)

```python
from geoscore import GeoScorer

# examples: list of dicts with 'question', 'correct', 'wrongs'
examples = [
    {
        "question": "What is the capital of France?",
        "correct":  "Paris",
        "wrongs":   ["London", "Berlin", "Madrid"]
    },
    # ... 49 more
]

scorer = GeoScorer("Qwen/Qwen2.5-1.5B-Instruct")
scorer.calibrate(examples)
scorer.save("my_task.pt")

# Later:
scorer = GeoScorer.load("Qwen/Qwen2.5-1.5B-Instruct", "my_task.pt")
```

---

## How it works

Three steps. No training at any step.

**1. Extract.** For each calibration example, compute the hidden state of the
correct answer and the mean hidden state of the wrong answers at ~65–70% of
model depth. Take the difference.

**2. Average.** Mean-pool these displacement vectors across 50 examples and
L2-normalise. This is the *correctness direction* δ̂.

**3. Score.** At inference, score each candidate answer by its dot product
with δ̂. Higher score → more aligned with the correctness geometry of the
model's internal representation.

The key finding: this direction is **task-specific**. A direction calibrated
on factual questions transfers to other factual tasks but not to commonsense
coherence or calibrated truthfulness tasks. Calibrate on 50 examples from
your target task for best results.

---

## Pre-computed directions

Ready-to-use directions for all five models in the paper, stored in
`directions/`. Load with `GeoScorer.from_pretrained(model, task)`.

| Model | `task="arc"` | `task="truthfulqa"` |
|-------|:---:|:---:|
| `meta-llama/Llama-3.2-1B-Instruct` | ✓ | ✓ |
| `Qwen/Qwen2.5-1.5B-Instruct` | ✓ | ✓ |
| `google/gemma-2-2b-it` | ✓ | ✓ |
| `Qwen/Qwen2.5-7B-Instruct` | ✓ | ✓ |
| `meta-llama/Llama-3.1-8B-Instruct` | ✓ | ✓ |

To generate directions for a model not listed here, see
[`scripts/precompute_directions.py`](scripts/precompute_directions.py).

---

## API reference

### `GeoScorer(model, layer_frac=None, device=None)`

Load a scorer for a given model. If `layer_frac` is not specified, the
optimal fraction for the model is used automatically (see paper Table A5).

### `GeoScorer.from_pretrained(model, task)`

Load a scorer with a pre-computed direction for `task` (`"arc"` or
`"truthfulqa"`). No calibration required.

### `GeoScorer.load(model, path)`

Load a scorer with a custom direction saved from a previous `calibrate()` call.

### `.calibrate(examples, layer_frac=None)`

Compute the contrastive correctness direction from a list of calibration
examples. Each example is a dict with keys `question` (str), `correct` (str),
and `wrongs` (list of str).

```python
examples = [{"question": ..., "correct": ..., "wrongs": [...]}, ...]
scorer.calibrate(examples)
```

### `.rank(question, choices)`

Score and rank a list of candidate answers. Returns a list of `(answer, score)`
tuples sorted by score descending.

```python
ranked = scorer.rank("What year did WWII end?", ["1945", "1944", "1946", "1943"])
# [("1945", 0.051), ("1944", 0.003), ("1946", -0.009), ("1943", -0.021)]
```

### `.confidence(question, answer)`

Return a scalar confidence score for a single (question, answer) pair.
Higher values indicate stronger alignment with the correctness direction.
Useful as a hallucination detection signal in production pipelines.

### `.save(path)` / `.load(model, path)`

Persist and reload a calibrated direction.

---

## Reproducing paper results

Each script in `examples/` reproduces the corresponding paper result and
prints a comparison against the reported numbers.

```bash
# ARC-Challenge — all five models
python examples/arc_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct

# TruthfulQA — task-specific calibration
python examples/truthfulqa_benchmark.py --model Qwen/Qwen2.5-7B-Instruct

# Hallucination detection AUROC
python examples/hallucination_detection.py --model Qwen/Qwen2.5-1.5B-Instruct

# Calibrate from scratch on a custom dataset
python examples/custom_calibration.py --help
```

Expected runtimes on a single A100 (all steps including calibration):
- 1B models: ~8 minutes per benchmark
- 7–8B models: ~25 minutes per benchmark

---

## Installation

```bash
pip install geoscore
```

Or from source:

```bash
git clone https://github.com/YOUR_USERNAME/geoscore
cd geoscore
pip install -e .
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, Transformers ≥ 4.40

---

## When to use geoscore

**Good fit:**
- Multiple-choice benchmarks and evaluation
- Retrieval reranking where candidates are format-consistent with calibration
- Hallucination flagging in structured Q&A pipelines (medical, legal, educational)
- Low-resource domain adaptation — 50 labeled examples replaces fine-tuning

**Not a good fit:**
- Open-ended generation where the generated text format differs from
  calibration examples (see §6.3 of the paper)
- Tasks requiring commonsense coherence rather than factual correctness
  (HellaSwag-style benchmarks perform inconsistently)

---

## Citation

```bibtex
@misc{armstrong2026correctnessdirectiongeometricanswer,
      title={Correctness Is a Direction: Geometric Answer Selection in Language Models}, 
      author={Marcus Armstrong and Navid Ayoobi and Pradham Mummaleti and Alexander Chulzhanov and Arjun Mukherjee},
      year={2026},
      eprint={2610.04512},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.04512}, 
}
```

---

## Paper

https://arxiv.org/abs/2610.04512

