"""RAGAS evaluation configuration — thresholds, LLM settings, and runner parameters.

These thresholds gate the pipeline evaluation (the primary regression gate).
Evaluation uses GPT-4o-mini for cost efficiency and reproducibility.
"""

import os

# Evaluation thresholds
THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
}

# Kept for historical reference; TARGET_THRESHOLDS is now equal to THRESHOLDS.
TARGET_THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
}

# Informational-only thresholds (not enforced in CI)
INFORMATIONAL_THRESHOLDS = {
    "context_recall": 0.65,
}

# Evaluator LLM (used by RAGAS to score metrics)
EVALUATOR_MODEL = os.environ.get("RAGAS_EVALUATOR_MODEL", "gpt-4o-mini")

# Eval runner settings
EVAL_USER_ROLE = "admin"
EVAL_USER_ID = "eval_runner"
