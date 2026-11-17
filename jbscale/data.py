"""Dataset loading: JBB-Behaviors and the Shen et al. DAN template.

No attack text is authored here. The DAN template is *selected* by index from a
released dataset of in-the-wild prompts; the harmful and benign requests come
straight from JBB-Behaviors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from datasets import load_dataset

from . import config


@dataclass(frozen=True)
class Behavior:
    """One request from JBB-Behaviors."""

    behavior_id: str
    split: str  # "harmful" or "benign"
    prompt: str
    category: str


def _rows(dataset_name: str, cfg: str):
    ds = load_dataset(dataset_name, cfg)
    # JBB exposes a single split per config; take whichever it is.
    split = next(iter(ds.keys()))
    return ds[split]


def _pick(row: dict, *names: str, default: str = "") -> str:
    for n in names:
        if n in row and row[n] is not None:
            return str(row[n])
    return default


def load_behaviors(split: str, limit: int | None = None) -> list[Behavior]:
    """Load harmful or benign JBB behaviors as a stable, ordered list."""
    if split == "harmful":
        rows = _rows(config.JBB_DATASET, config.JBB_HARMFUL_CONFIG)
    elif split == "benign":
        rows = _rows(config.JBB_DATASET, config.JBB_BENIGN_CONFIG)
    else:
        raise ValueError(f"Unknown split {split!r}")

    behaviors: list[Behavior] = []
    for i, row in enumerate(rows):
        prompt = _pick(row, "Goal", "goal", "Behavior", "behavior", "prompt")
        if not prompt:
            continue
        # Prefer the dataset's own index so ids line up across harmful/benign.
        idx = _pick(row, "Index", "index", default=str(i))
        behaviors.append(
            Behavior(
                behavior_id=f"{split}_{idx}",
                split=split,
                prompt=prompt,
                category=_pick(row, "Category", "category", default="unknown"),
            )
        )
        if limit is not None and len(behaviors) >= limit:
            break
    return behaviors


@lru_cache(maxsize=1)
def dan_template() -> str:
    """The single in-the-wild DAN template, selected by the configured index.

    Returns a format string containing exactly one ``{prompt}`` placeholder
    where the harmful request is inserted. The raw community prompt has no such
    placeholder, so we append the request after the role-play instructions,
    which is how these templates are used in the wild.
    """
    token = os.environ.get(config.HF_TOKEN_ENV)
    ds = load_dataset(
        config.DAN_DATASET,
        config.DAN_DATASET_CONFIG,
        split=config.DAN_DATASET_SPLIT,
        token=token,
    )
    # Keep only genuine jailbreak prompts, ordered stably for reproducibility.
    def is_jb(r):
        v = r.get("jailbreak")
        return bool(v) if v is not None else True

    rows = [r for r in ds if is_jb(r)]
    sort_key = "community_id" if rows and "community_id" in rows[0] else "prompt"
    rows.sort(key=lambda r: str(r.get(sort_key, "")))
    if not rows:
        raise RuntimeError("No DAN prompts found in the configured dataset split.")
    chosen = rows[config.DAN_TEMPLATE_INDEX % len(rows)]
    base = str(chosen["prompt"]).strip()
    return base + "\n\n{prompt}"
