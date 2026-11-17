"""Orchestration: run attacks across behaviors and sizes, grade, and store.

Keeps the generation notebook to a few calls. Everything here is resumable via
``ResultStore`` and logs attacker/attack failures rather than scoring them as
zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import attacks, config
from .data import Behavior, load_behaviors
from .judge import grade
from .store import Record, ResultStore
from .targets import TargetClient


@dataclass
class PapRewrite:
    behavior_id: str
    technique: str
    rewrite: str
    failed: bool


def prepare_pap_rewrites(behaviors: list[Behavior], store_path: str | None = None) -> dict:
    """Make and cache PAP rewrites once, keyed by (behavior_id, technique).

    Rewrites are target-independent, so they are generated a single time and
    reused at every size, exactly as the proposal requires.
    """
    import json
    import os

    path = store_path or config.PATHS.pap_cache
    os.makedirs(os.path.dirname(path), exist_ok=True)

    cache: dict[tuple[str, str], PapRewrite] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cache[(r["behavior_id"], r["technique"])] = PapRewrite(**r)

    with open(path, "a", encoding="utf-8") as f:
        for behavior in behaviors:
            for technique in config.PAP_TECHNIQUES:
                if (behavior.behavior_id, technique) in cache:
                    continue
                rewrite, failed = attacks.make_pap_rewrite(behavior.prompt, technique)
                rec = PapRewrite(behavior.behavior_id, technique, rewrite, failed)
                cache[(behavior.behavior_id, technique)] = rec
                f.write(json.dumps(rec.__dict__) + "\n")
                f.flush()
    return cache


def _store_one(store: ResultStore, behavior: Behavior, target: config.TargetModel,
               attack: str, outcome: attacks.AttackOutcome) -> None:
    if outcome.attack_failed:
        # Attack never reached the target usefully: record as missing, not zero.
        store.append(Record(
            behavior_id=behavior.behavior_id, split=behavior.split,
            model=target.key, n_params=target.n_params, attack=attack,
            strongreject=float("nan"), refused=False, attack_failed=True,
            variant=outcome.variant,
        ))
        return
    judged = grade(behavior.prompt, outcome.response)
    store.append(Record(
        behavior_id=behavior.behavior_id, split=behavior.split,
        model=target.key, n_params=target.n_params, attack=attack,
        strongreject=judged.score, refused=judged.refused, attack_failed=False,
        variant=outcome.variant, judge_model=judged.judge_model,
    ))


def run_model(target_model: config.TargetModel, behaviors: list[Behavior],
              benign: list[Behavior], pap_cache: dict, store: ResultStore,
              arms: tuple[str, ...] | None = None) -> None:
    """Run every configured arm for one target size and append graded rows."""
    arms = arms or config.ATTACKS
    client = TargetClient(target_model)

    for behavior in behaviors:
        for attack in arms:
            if attack == "pap":
                for technique in config.PAP_TECHNIQUES:
                    if store.has(behavior.behavior_id, target_model.key, attack, technique):
                        continue
                    pr = pap_cache.get((behavior.behavior_id, technique))
                    if pr is None:
                        rewrite, failed = attacks.make_pap_rewrite(behavior.prompt, technique)
                    else:
                        rewrite, failed = pr.rewrite, pr.failed
                    outcome = attacks.run_pap(behavior, client, rewrite, failed, technique)
                    _store_one(store, behavior, target_model, attack, outcome)
                continue

            if store.has(behavior.behavior_id, target_model.key, attack):
                continue
            if attack == "none":
                outcome = attacks.run_none(behavior, client)
            elif attack == "dan":
                outcome = attacks.run_dan(behavior, client)
            else:  # crescendo or pair
                outcome = attacks.run_adaptive(behavior, client)
            _store_one(store, behavior, target_model, attack, outcome)

    # Benign baseline: no-attack only, for over-refusal.
    for behavior in benign:
        if store.has(behavior.behavior_id, target_model.key, "none"):
            continue
        outcome = attacks.run_none(behavior, client)
        _store_one(store, behavior, target_model, "none", outcome)


def attacker_refusal_rate(pap_cache: dict) -> float:
    """Fraction of PAP rewrites the attacker refused, for the run manifest."""
    if not pap_cache:
        return float("nan")
    failed = sum(1 for r in pap_cache.values() if r.failed)
    return failed / len(pap_cache)
