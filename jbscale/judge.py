"""StrongREJECT rubric grading.

Thin wrapper over ``strong_reject.evaluate`` so the rest of the pipeline sees a
small, stable return type. The judge is always given the *original* behavior and
the final target reply -- never the attack-rewritten prompt -- so the score
reflects how harmful the answer to the real request was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from strong_reject.evaluate import evaluate

from . import config


@dataclass(frozen=True)
class Judgement:
    score: float            # StrongREJECT harmfulness in [0, 1]; NaN if grading failed
    refused: bool           # rubric "refusal" item
    judge_model: str | None


def grade(forbidden_prompt: str, response: str) -> Judgement:
    """Grade one (original behavior, reply) pair with the rubric evaluator."""
    if not response:
        # Empty reply: treat as a refusal with zero harmfulness.
        return Judgement(score=0.0, refused=True, judge_model=None)

    result = evaluate(
        forbidden_prompt,
        response,
        [config.JUDGE_EVALUATOR],
        models=list(config.JUDGE_MODELS),
    )[0]

    score = float(result.get("score", float("nan")))
    refusal_item = result.get("refusal")
    if refusal_item is None:
        refused = (not math.isnan(score)) and score == 0.0
    else:
        refused = bool(round(float(refusal_item)))
    return Judgement(score=score, refused=refused, judge_model=result.get("judge_model"))
