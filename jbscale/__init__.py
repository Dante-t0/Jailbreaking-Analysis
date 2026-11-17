"""jbscale: generation pipeline for the jailbreak-scale study.

Submodules:
    config   static run configuration (models, attacks, judge, paths)
    data     JBB-Behaviors and the Shen et al. DAN template
    targets  Together OpenAI-compatible clients for Qwen targets / attacker
    attacks  adapters over published DAN / PAP / Crescendo / PAIR
    judge    StrongREJECT rubric grading
    store    resumable JSONL record store + scores.csv export
    runner   orchestration across behaviors and sizes

Importing this package is cheap; heavy deps (torch, pyrit) load lazily only
when an attack or the judge actually runs.
"""

from . import config  # noqa: F401

__all__ = ["config"]
