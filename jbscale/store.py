"""Append-only JSONL result store with resume and CSV export.

Every graded (behavior, split, model, attack) cell is one JSON line in
``results/raw/records.jsonl``. Reruns skip keys already present, so a run that
dies partway through resumes where it left off. ``export_scores`` flattens the
records into the long ``results/scores.csv`` that the analysis notebook reads.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from . import config


@dataclass
class Record:
    behavior_id: str
    split: str          # "harmful" | "benign"
    model: str          # target key, e.g. "7b"
    n_params: int
    attack: str         # none | dan | pap | crescendo | pair
    strongreject: float
    refused: bool
    attack_failed: bool
    variant: str = ""   # PAP technique when more than one is configured
    judge_model: str | None = None


def _key(behavior_id: str, model: str, attack: str, variant: str = "") -> tuple[str, str, str, str]:
    return (behavior_id, model, attack, variant)


class ResultStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.path.join(config.PATHS.raw, "records.jsonl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._done: set[tuple[str, str, str, str]] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                self._done.add(_key(r["behavior_id"], r["model"], r["attack"], r.get("variant", "")))

    def has(self, behavior_id: str, model: str, attack: str, variant: str = "") -> bool:
        return _key(behavior_id, model, attack, variant) in self._done

    def append(self, record: Record) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        self._done.add(_key(record.behavior_id, record.model, record.attack, record.variant))

    def records(self) -> list[dict]:
        out: list[dict] = []
        if not os.path.exists(self.path):
            return out
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def export_scores(self, csv_path: str | None = None) -> str:
        """Write the long scores CSV. Returns the path written."""
        import pandas as pd

        csv_path = csv_path or config.PATHS.scores_csv
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        rows = self.records()
        cols = [
            "behavior_id", "split", "model", "n_params",
            "attack", "variant", "strongreject", "refused", "attack_failed",
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df[cols + [c for c in df.columns if c not in cols]]
        df.to_csv(csv_path, index=False)
        return csv_path
