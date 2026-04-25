# Confidence-Aware Medical Diagnostic System

A medical diagnostic system built on **Llama-3.2-3B-Instruct** that can provide patients with a safe, surface-level diagnosis. The system knows when to diagnose and when to refer to a specialist.

Built and evaluated using the [DDXPlus dataset](https://huggingface.co/datasets/aai530-group6/ddxplus) — 1.3M+ synthetic patient cases across 49 diseases.

---

## How It Works

The system works in two steps:

1. **The model diagnoses** — Llama analyzes patient symptoms, age, and sex alongside a list of possible diseases, and returns its top predictions with confidence scores.

2. **The code checks safety** — A decision layer in the code applies two rules before any result reaches the user:
   - If the predicted disease is **severe** (e.g., pulmonary embolism, anaphylaxis) → automatically refer to a specialist
   - If the model's confidence is **below a dynamic threshold** (`2 / N` where N = number of possible diseases) → refer to a specialist

The model only handles diagnosis. The code guarantees safety.

---

## Phases

| Phase | Description | Model | Decision Layer |
|-------|-------------|-------|----------------|
| **1a** | Baseline — raw diagnostic ability | Untrained | None |
| **1b** | Baseline + safety rules in prompt | Untrained | In prompt |
| **2** | Fine-tuned + safety rules in prompt | Trained on DDXPlus | In prompt |
| **3** | Fine-tuned + safety rules in code | Trained on DDXPlus | In code |

---

## Project Structure

├── datasets/EDA              # Dataset exploration and analysis
│
├── src/
│   ├── baseline.py                        # Phase 1a: raw model, no prompting
│   ├── baseline_with_system_prompt.py     # Phase 1b & 2: decision rules in prompt
│   └── decision_layer.py                  # Phase 3: decision rules in code
│
├── evaluation/
│   └── evaluation.py                      # Pass@1, F1, precision, recall
│
├── results/
│   ├── phase1a/
│   │   ├── phase1a_results.csv
│   │   └── plots/
│   ├── phase1b/
│   │   ├── phase1b_results.csv
│   │   └── plots/
│   ├── phase2/
│   │   ├── phase2_results.csv
│   │   └── plots/
│   └── phase3/
│       ├── phase3_results.csv
│       └── plots/

---

## Evaluation

We use **strict Pass@1 accuracy** where a case is only counted as correct if:

- The diagnosis is right **AND** confidence exceeds the dynamic threshold
- Severe diseases are properly referred (not diagnosed)
- Correct referrals (severe cases referred, low-confidence cases referred) also count as successes

Additional metrics: **Precision**, **Recall**, and **F1 Score**.

---

## Decision Layer

The safety layer applies two checks to every prediction:

**Severe disease check** — 22 high-risk diseases (e.g., pulmonary embolism, anaphylaxis, unstable angina, tuberculosis) are always referred to a specialist, regardless of confidence.

**Dynamic confidence threshold** — The model must be at least 2x more confident than random chance. For a case with 5 possible diseases, the threshold is 40%. For 10 diseases, it's 20%. This scales with case complexity instead of using a fixed cutoff.

If both checks pass → the system shows the diagnosis.
If either fails → the system refers to a specialist.

---

## Dataset

**DDXPlus** — a large-scale synthetic medical dataset presented at NeurIPS 2022.

- 1.3M+ patient cases
- 49 diseases
- Each case includes: age, sex, symptoms, evidences, differential diagnosis with probabilities, and ground truth pathology

---

## Tech Stack

- **Model**: Llama-3.2-3B-Instruct (Meta)
- **Dataset**: DDXPlus (aai530-group6/ddxplus)
- **Framework**: PyTorch, Hugging Face Transformers
- **Evaluation**: scikit-learn, pandas, matplotlib, seaborn
