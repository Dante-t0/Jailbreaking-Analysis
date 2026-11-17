"""Attack adapters.

Each arm is wrapped behind one small function returning an ``AttackOutcome``
(the final target reply to be graded, plus whether the attack itself failed).
We author no attack text: DAN is a selected template, PAP comes from
``strong_reject``, and the multi-turn arm is PyRIT's Crescendo (or
``strong_reject``'s PAIR as the proposal's fallback).

Heavy dependencies (torch via strong_reject, pyrit) are imported lazily inside
functions so that ``import jbscale`` stays cheap and the analysis notebook never
pulls them in.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from . import config
from .data import Behavior, dan_template
from .targets import TargetClient, together_client


@dataclass
class AttackOutcome:
    response: str          # final target reply, to be graded by StrongREJECT
    attack_failed: bool    # attacker refused / rewrite failed / attack errored
    variant: str = ""      # PAP technique, when more than one is configured
    detail: str = ""


# --------------------------------------------------------------------------- #
# Single-turn arms.
# --------------------------------------------------------------------------- #

def run_none(behavior: Behavior, target: TargetClient) -> AttackOutcome:
    """No-attack baseline: the behavior as written."""
    return AttackOutcome(response=target.complete_prompt(behavior.prompt), attack_failed=False)


def run_dan(behavior: Behavior, target: TargetClient) -> AttackOutcome:
    """Fixed in-the-wild DAN template with the request appended."""
    prompt = dan_template().format(prompt=behavior.prompt)
    return AttackOutcome(response=target.complete_prompt(prompt), attack_failed=False)


# --------------------------------------------------------------------------- #
# PAP: persuasion rewrite, made once and reused across sizes.
# --------------------------------------------------------------------------- #

def make_pap_rewrite(behavior_prompt: str, technique: str) -> tuple[str, bool]:
    """Rewrite a request with one PAP technique using the attacker model.

    Returns (rewritten_prompt, failed). ``failed`` is True when the attacker
    refused or produced nothing usable; ``strong_reject.pap`` signals this by
    returning the original prompt unchanged.
    """
    from strong_reject.jailbreaks import pap  # lazy: pulls torch

    rewrite = pap(
        behavior_prompt,
        template=technique,
        models=(config.ATTACKER_MODEL_LITELLM,),
        max_tokens=config.TARGET_MAX_TOKENS,
    )
    failed = (rewrite or "").strip() == behavior_prompt.strip() or not rewrite.strip()
    return rewrite, failed


def run_pap(behavior: Behavior, target: TargetClient, rewrite: str, failed: bool,
            technique: str = "") -> AttackOutcome:
    """Send a cached PAP rewrite to the target."""
    if failed:
        return AttackOutcome(response="", attack_failed=True, variant=technique,
                             detail="attacker failed to produce a PAP rewrite")
    return AttackOutcome(response=target.complete_prompt(rewrite), attack_failed=False,
                         variant=technique)


# --------------------------------------------------------------------------- #
# Multi-turn: Crescendo (PyRIT) with PAIR (strong_reject) as fallback.
# --------------------------------------------------------------------------- #

_PYRIT_READY = False


def ensure_pyrit() -> None:
    """Initialise PyRIT's in-memory store once per process."""
    global _PYRIT_READY
    if _PYRIT_READY:
        return
    from pyrit.setup import IN_MEMORY, initialize_pyrit_async

    asyncio.get_event_loop().run_until_complete(initialize_pyrit_async(memory_db_type=IN_MEMORY))
    _PYRIT_READY = True


def _pyrit_target(model_string: str, temperature: float | None) -> "object":
    from pyrit.prompt_target import OpenAIChatTarget
    from .targets import _require

    return OpenAIChatTarget(
        model_name=model_string,
        endpoint=config.TOGETHER_BASE_URL,
        api_key=_require(config.TOGETHER_API_KEY_ENV),
        temperature=temperature,
        max_completion_tokens=config.TARGET_MAX_TOKENS,
    )


def run_crescendo(behavior: Behavior, target: TargetClient) -> AttackOutcome:
    """PyRIT Crescendo: escalate over turns, then grade the final reply."""
    from pyrit.executor.attack import AttackAdversarialConfig, CrescendoAttack
    from pyrit.models import AttackOutcome as PyritOutcome

    ensure_pyrit()
    attacker = _pyrit_target(config.ATTACKER_MODEL, temperature=None)
    objective_target = _pyrit_target(target.model_string, temperature=config.TARGET_TEMPERATURE)

    attack = CrescendoAttack(
        objective_target=objective_target,
        attack_adversarial_config=AttackAdversarialConfig(target=attacker),
        max_turns=config.CRESCENDO_MAX_TURNS,
        max_backtracks=config.CRESCENDO_MAX_BACKTRACKS,
    )
    result = asyncio.get_event_loop().run_until_complete(
        attack.execute_async(objective=behavior.prompt)
    )

    last = getattr(result, "last_response", None)
    reply = getattr(last, "converted_value", "") if last is not None else ""
    failed = result.outcome == PyritOutcome.ERROR or not reply
    return AttackOutcome(response=reply, attack_failed=failed,
                         detail=f"outcome={result.outcome}, turns={result.executed_turns}")


def run_pair(behavior: Behavior, target: TargetClient) -> AttackOutcome:
    """strong_reject PAIR fallback: search for a jailbreak prompt, then grade."""
    from strong_reject.jailbreaks import pair  # lazy: pulls torch

    victim_litellm = f"together_ai/{target.model_string}"
    jailbroken_prompt = pair(
        behavior.prompt,
        attack_models=(config.ATTACKER_MODEL_LITELLM,),
        victim_model=victim_litellm,
        max_iter=config.PAIR_MAX_ITER,
        max_conversation_length=config.PAIR_MAX_CONVERSATION_LENGTH,
    )
    failed = (jailbroken_prompt or "").strip() == behavior.prompt.strip()
    reply = target.complete_prompt(jailbroken_prompt)
    return AttackOutcome(response=reply, attack_failed=failed)


def run_adaptive(behavior: Behavior, target: TargetClient) -> AttackOutcome:
    """Dispatch the third arm according to ``config.ADAPTIVE_ARM``."""
    if config.ADAPTIVE_ARM == "pair":
        return run_pair(behavior, target)
    return run_crescendo(behavior, target)
