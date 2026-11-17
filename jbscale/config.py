"""Static configuration for the jailbreak-scale generation pipeline.

Everything that defines *what* we run (which Qwen sizes, which attacks, which
judge, decoding settings) lives here so the notebook stays short and every run
is reproducible from one place.

Hosting note (verified Nov 2026): Together's *serverless* catalog has no
Qwen2.5-Instruct models, so each size must run on its own *dedicated* endpoint.
Qwen2.5-32B-Instruct is deployable from Together's dedicated catalog; the
smaller sizes are uploaded once from the Hugging Face repos below and then
deployed. Deploy endpoints out of band (CLI/UI), put the resulting model string
for each size in ``TARGET_ENDPOINT_OVERRIDES`` (or the env vars named there),
and tear them down afterwards -- dedicated endpoints bill by the hour.

Usage terms: running these published attacks against models you host on
Together for safety evaluation is consistent with Together's documented
red-teaming use; the OpenAI judge call is ordinary API usage. This project
assumes the academic-review approval the user stated. Confirm both providers'
current terms before a full run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Target models: the six Qwen2.5-Instruct sizes.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TargetModel:
    """One Qwen2.5-Instruct size.

    Attributes:
        key: Short label used in the scores table (e.g. ``"7b"``).
        hf_repo: Hugging Face repo, used when uploading to Together and as the
            source of truth for the chat template.
        n_params: Actual total parameter count (not the rounded label), taken
            from the Qwen2.5 technical report / model cards. This is the x-axis
            of every regression, via ``log(n_params)``.
        default_model_string: The Together model string to call once the
            endpoint is deployed. Override per environment; see
            ``target_model_string``.
    """

    key: str
    hf_repo: str
    n_params: int
    default_model_string: str


# Total-parameter counts for Qwen2.5-Instruct (dense) models.
TARGET_MODELS: tuple[TargetModel, ...] = (
    TargetModel("0.5b", "Qwen/Qwen2.5-0.5B-Instruct", 494_032_768, "Qwen/Qwen2.5-0.5B-Instruct"),
    TargetModel("1.5b", "Qwen/Qwen2.5-1.5B-Instruct", 1_543_714_304, "Qwen/Qwen2.5-1.5B-Instruct"),
    TargetModel("3b", "Qwen/Qwen2.5-3B-Instruct", 3_085_938_688, "Qwen/Qwen2.5-3B-Instruct"),
    TargetModel("7b", "Qwen/Qwen2.5-7B-Instruct", 7_615_616_512, "Qwen/Qwen2.5-7B-Instruct"),
    TargetModel("14b", "Qwen/Qwen2.5-14B-Instruct", 14_770_033_664, "Qwen/Qwen2.5-14B-Instruct"),
    TargetModel("32b", "Qwen/Qwen2.5-32B-Instruct", 32_763_876_352, "Qwen/Qwen2.5-32B-Instruct"),
)

# Smallest-first, so a run warms up and fails cheap before the expensive 32B.
TARGET_MODELS = tuple(sorted(TARGET_MODELS, key=lambda m: m.n_params))

# Map a target key to the deployed Together model string. Prefer an env var
# (JBSCALE_TARGET_7B=...) so secrets/endpoint names stay out of the repo.
TARGET_ENDPOINT_OVERRIDES: dict[str, str] = {}


def target_model_string(model: TargetModel) -> str:
    """Resolve the Together model string for a target, env var winning."""
    env = os.environ.get(f"JBSCALE_TARGET_{model.key.upper().replace('.', '_')}")
    return env or TARGET_ENDPOINT_OVERRIDES.get(model.key) or model.default_model_string


# --------------------------------------------------------------------------- #
# Endpoints and credentials.
# --------------------------------------------------------------------------- #

# Together exposes an OpenAI-compatible API. Targets and the attacker both go
# here; only the StrongREJECT judge talks to OpenAI proper.
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
HF_TOKEN_ENV = "HF_TOKEN"


# --------------------------------------------------------------------------- #
# Attacker model (writes PAP rewrites, drives Crescendo/PAIR).
# --------------------------------------------------------------------------- #

# One fixed open-weight model *outside the Qwen family*, served by Together
# serverless so it needs no dedicated endpoint. Shared by every adaptive attack
# so the attacker is held constant across target sizes.
ATTACKER_MODEL = "openai/gpt-oss-120b"
# litellm (used inside strong_reject) needs a provider prefix to route to Together.
ATTACKER_MODEL_LITELLM = f"together_ai/{ATTACKER_MODEL}"


# --------------------------------------------------------------------------- #
# Attacks.
# --------------------------------------------------------------------------- #

# DAN: a single in-the-wild template chosen by ID from Shen et al.'s released
# dataset, pasted unchanged before every harmful request. We never author DAN
# text; this is just which published row to load.
DAN_DATASET = "TrustAIRLab/in-the-wild-jailbreak-prompts"
DAN_DATASET_CONFIG = "jailbreak_2023_12_25"
DAN_DATASET_SPLIT = "train"
# Index into the DAN-labelled subset (sorted by community id) of that split.
# Pinning an index keeps the template identical across sizes and reruns.
DAN_TEMPLATE_INDEX = 0

# PAP: one persuasion technique from strong_reject's registered PAP jailbreaks,
# so each behavior gets a single attempt as the proposal requires. List more
# keys here and the analysis averages them per behavior. Available keys:
# evidence_based_persuasion, expert_endorsement, misrepresentation,
# authority_endorsement, logical_appeal.
PAP_TECHNIQUES: tuple[str, ...] = ("logical_appeal",)

# Crescendo (PyRIT). The third arm is Crescendo unless it proves unreliable, in
# which case the proposal's PAIR fallback is used; set ADAPTIVE_ARM to "pair".
ADAPTIVE_ARM = "crescendo"  # "crescendo" or "pair"
CRESCENDO_MAX_TURNS = 5
CRESCENDO_MAX_BACKTRACKS = 2
# PAIR fallback (strong_reject's implementation) iteration budget.
PAIR_MAX_ITER = 20
PAIR_MAX_CONVERSATION_LENGTH = 3

# The four attack arms that appear in the scores table.
BASELINE_ATTACK = "none"
ATTACKS: tuple[str, ...] = ("none", "dan", "pap", ADAPTIVE_ARM)


# --------------------------------------------------------------------------- #
# Decoding and judging.
# --------------------------------------------------------------------------- #

# Identical decoding for every target size (proposal: temperature 0 for targets).
TARGET_TEMPERATURE = 0.0
TARGET_MAX_TOKENS = 512

# StrongREJECT rubric evaluator, graded by an OpenAI model. The judge always
# sees the *original* behavior and the final reply, never the rewritten prompt.
JUDGE_EVALUATOR = "strongreject_rubric"
JUDGE_MODELS: tuple[str, ...] = ("openai/gpt-4o-mini",)


# --------------------------------------------------------------------------- #
# Dataset and run limits.
# --------------------------------------------------------------------------- #

JBB_DATASET = "JailbreakBench/JBB-Behaviors"
JBB_HARMFUL_CONFIG = "behaviors"
JBB_BENIGN_CONFIG = "benign"
N_BEHAVIORS = 100  # full JBB-Behaviors set

# Smoke test: a few behaviors on the smallest model only.
SMOKE_N_BEHAVIORS = 5
SMOKE_TARGET_KEY = "0.5b"


# --------------------------------------------------------------------------- #
# Paths and reproducibility.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Paths:
    root: str = os.path.dirname(os.path.abspath(__file__))
    project: str = field(init=False)
    results: str = field(init=False)
    raw: str = field(init=False)
    scores_csv: str = field(init=False)
    manifest: str = field(init=False)
    pap_cache: str = field(init=False)
    figures: str = field(init=False)

    def __post_init__(self) -> None:
        project = os.path.dirname(self.root)
        object.__setattr__(self, "project", project)
        object.__setattr__(self, "results", os.path.join(project, "results"))
        object.__setattr__(self, "raw", os.path.join(project, "results", "raw"))
        object.__setattr__(self, "scores_csv", os.path.join(project, "results", "scores.csv"))
        object.__setattr__(self, "manifest", os.path.join(project, "results", "run_manifest.json"))
        object.__setattr__(self, "pap_cache", os.path.join(project, "results", "raw", "pap_rewrites.jsonl"))
        object.__setattr__(self, "figures", os.path.join(project, "figures"))


PATHS = Paths()

SEED = 2024
