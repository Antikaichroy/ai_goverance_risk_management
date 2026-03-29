# PII Risk & Hallucination Scoring

[![Python](https://img.shields.io/badge/Python-3.10.18-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-31018/)
[![GLiNER](https://img.shields.io/badge/GLiNER-nvidia%2Fgliner--PII-green)](https://huggingface.co/nvidia/gliner-PII)
[![NumPy](https://img.shields.io/badge/NumPy-supported-lightgrey?logo=numpy)](https://numpy.org)
[![CUDA](https://img.shields.io/badge/Device-CUDA%20%7C%20CPU-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

A lightweight pipeline for detecting personally identifiable information (PII) in text and computing a quantitative risk score, alongside a configurable hallucination scoring utility for RAG-based systems.

---

## Overview

This module provides two independent scoring functions:

- **PII Risk Scorer** — Uses a pre-trained NER model to identify PII entities in a given text, maps each entity to a risk class, and computes a composite risk score that accounts for both the severity of detected entities and their cumulative exposure.
- **Hallucination Scorer** — Computes a composite hallucination score using a geometric mean of three RAG evaluation metrics: context accuracy, faithfulness, and answer relevance.

---

## Dependencies

- Python 3.10.18
- [`gliner`](https://github.com/urchade/GLiNER) — Named entity recognition model
- `numpy`
- A local module `sample_sentences` providing `pii_sentences` and `non_pii_sentences` for testing

Install the required packages:

```bash
pip install gliner numpy
```

---

## Model

The pipeline uses the `nvidia/gliner-PII` model loaded via the GLiNER framework. By default it targets a CUDA device. To run on CPU, modify the `map_location` argument:

```python
extractor = GLiNER.from_pretrained("nvidia/gliner-PII", map_location="cpu")
```

---

## PII Labels

The following entity types are detected:

| Label | Description |
|---|---|
| `person name` | Full or partial name of an individual |
| `address` | Physical address |
| `license number` | Driver's license or similar ID number |
| `bank` | Bank account or institution reference |
| `phone` | Phone number |
| `numerical salary amount` | Exact salary figures |
| `personal location` | Geographic location tied to an individual |
| `job role` | Occupation or job title |
| `height` | Physical height |
| `weight` | Physical weight |
| `credit card number` | Payment card number |

---

## Risk Class Mapping

Each detected label is assigned an importance weight:

| Weight | Labels |
|---|---|
| `1.0` (High) | credit card number, personal location, address, phone, bank, license number |
| `0.5` (Medium) | height, weight, numerical salary amount, job role |
| `0.0` (Low) | person name |

If the same entity class is detected more than once, its risk value is doubled to reflect repeated exposure.

---

## API Reference

### `risk_class_val(res)`

Extracts risk class values from GLiNER entity predictions. Only entities with a confidence score of `0.7` or above are considered.

**Parameters**
- `res` (`List[Dict]`) — Raw output from `extractor.predict_entities()`

**Returns**
- `List[float]` — A list of risk weights for all detected (and sufficiently confident) PII entities

---

### `calculate_risk_score(risk_class_values, alpha)`

Computes a normalized risk score in the range `[0.0, 1.0]`.

**Parameters**
- `risk_class_values` (`List[float]`) — Output of `risk_class_val()`
- `alpha` (`float`) — Weighting factor for the maximum risk value vs. cumulative exposure. A higher alpha prioritizes worst-case severity; a lower alpha emphasizes breadth of exposure.

**Formula**

```
exposure     = mean_risk * log(1 + N)
risk_score   = min(1.0, alpha * max_risk + (1 - alpha) * exposure)
```

**Returns**
- `float` — Risk score rounded to 2 decimal places. Returns `0.0` if no entities are detected.

---

### `calculate_hallucination_score(...)`

Computes a weighted hallucination probability from three RAG quality metrics.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `CA` | `float` | Score for context accuracy (0–1) |
| `FA` | `float` | Score for answer faithfulness to context (0–1) |
| `AR` | `float` | Score for relevance of the answer to the query (0–1) |
| `alpha` | `float` | Sensitivity for context accuracy |
| `beta` | `float` | Sensitivity for faithfulness |
| `gamma` | `float` | Sensitivity for answer relevance |

alpha + beta + gamma = 1

alpha, beta, gamma:  Exponent-based sensitivities for CA, FA, and AR to manage their relative impact on the overall score. Increasing the exponent will make the score more sensitive to changes in that measure. 

**Formula**

```
hallucination_score = (CA ** alpha) * (FA ** beta) * (AR ** gamma)
H = round(1 - hallucination_score, 2)

```

**Returns**
- `float` — Hallucination score rounded to 2 decimal places.

---

## Usage Example

```python
from gliner import GLiNER
from example_score_cal import calculate_risk_score, calculate_hallucination_score, risk_class_val

extractor = GLiNER.from_pretrained("nvidia/gliner-PII", map_location="cpu")

text = "John's credit card number is 4111 1111 1111 1111 and he lives at 42 Elm Street."
res = extractor.predict_entities(text, labels=labels)

risk_values = risk_class_val(res)
risk = calculate_risk_score(risk_values, alpha=0.6)
print(f"PII Risk Score: {risk}")

hallucination = calculate_hallucination_score(
    CA=0.6,
    FA=0.2,
    AR=0.4,
    alpha=0.4,
    beta=0.4,
    gamma=0.2,
)
print(f"Hallucination Score: {hallucination}")
```

---

## Notes

- The `alpha` parameter in `calculate_risk_score` controls the trade-off between peak severity and breadth of exposure. A value of `0.6` is a reasonable default, weighting worst-case entities more heavily.
- alpha, beta, gamma in `calculate_hallucination_score` should sum to `1.0` for the score to remain within a consistent interpretable range.
- The confidence threshold of `0.7` in `risk_class_val` filters out low-confidence entity predictions and can be adjusted based on the acceptable false-positive rate for a given application.