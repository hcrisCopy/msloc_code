"""Crash-tolerant per-rank JSONL audit logs for OPD and GRPO rollouts."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set


class RolloutAuditWriter:
    """Append rollout records without making distributed ranks share a file.

    A stable ``audit_key`` makes a checkpoint resume idempotent.  Each append is
    flushed before returning.  If a process was interrupted halfway through its
    final JSON line, initialization keeps all preceding valid lines and removes
    only that incomplete tail before appending new records.
    """

    def __init__(self, output_dir: str, stage: str, rank: int, enabled: bool):
        self.enabled = bool(enabled)
        self.stage = str(stage)
        self.rank = int(rank)
        self.path = Path(output_dir) / f"{self.stage}_rank{self.rank:05d}.jsonl"
        self._keys: Set[str] = set()
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load_and_repair()

    def _load_and_repair(self) -> None:
        if not self.path.exists():
            return
        valid_lines = []
        damaged_tail = False
        missing_final_newline = False
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    damaged_tail = True
                    break
                key = record.get("audit_key")
                if isinstance(key, str) and key:
                    self._keys.add(key)
                if not line.endswith("\n"):
                    missing_final_newline = True
                valid_lines.append(line if line.endswith("\n") else line + "\n")
        if damaged_tail or missing_final_newline:
            temporary = self.path.with_suffix(self.path.suffix + ".repair.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.writelines(valid_lines)
                handle.flush()
            temporary.replace(self.path)

    def write(self, records: Iterable[Dict[str, Any]]) -> int:
        if not self.enabled:
            return 0
        pending = []
        for record in records:
            key = record.get("audit_key")
            if not isinstance(key, str) or not key:
                raise ValueError("Every rollout audit record needs a non-empty audit_key")
            if key in self._keys:
                continue
            pending.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            self._keys.add(key)
        if not pending:
            return 0
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(pending) + "\n")
            handle.flush()
        return len(pending)
