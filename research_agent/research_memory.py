"""Cross-run, append-only evidence index for autonomous research planning.

The index is deliberately a *copy* of compact evidence from run artifacts.
It never edits or replaces the source logs, and it is not a catalogue of
allowed experiments.  Its purpose is to let a fresh run reason from prior
results instead of rediscovering the same failed directions.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


class ResearchMemory:
    """Maintain a durable, de-duplicated index of completed research evidence."""

    def __init__(self, workspace_root: str | Path, root: str | Path | None = None) -> None:
        self.workspace_root = Path(workspace_root)
        self.root = Path(root) if root is not None else self.workspace_root / "research_memory"
        self.archive_path = self.root / "experiment_archive.jsonl"

    def bootstrap(self, *, exclude_run: str | Path | None = None) -> dict[str, Any]:
        """Import existing run evidence once, without changing its source files."""
        excluded = Path(exclude_run).resolve() if exclude_run is not None else None
        imported = 0
        sources: list[str] = []
        for run_dir in sorted(self.workspace_root.iterdir()):
            if not run_dir.is_dir() or not (run_dir / "iterations.jsonl").exists():
                continue
            if excluded is not None and run_dir.resolve() == excluded:
                continue
            before = len(self.records())
            self.ingest_run(run_dir)
            if len(self.records()) > before:
                imported += len(self.records()) - before
                sources.append(run_dir.name)
        return {"imported_records": imported, "source_runs": sources, "total_records": len(self.records())}

    def ingest_run(self, run_dir: str | Path) -> int:
        """Append compact copies of iterations and material candidate failures."""
        directory = Path(run_dir)
        source_run = directory.name
        existing = self._keys()
        pending: list[dict[str, Any]] = []
        for record in _read_jsonl(directory / "iterations.jsonl"):
            item = _iteration_record(source_run, record)
            if item["memory_key"] not in existing:
                pending.append(item)
                existing.add(item["memory_key"])
        for event in _read_jsonl(directory / "experiments.jsonl"):
            if event.get("action") not in {
                "candidate_failure_recorded", "isolated_candidate_rejected", "candidate_abandoned",
                "safety_rejected",
            }:
                continue
            item = _failure_event(source_run, event)
            if item["memory_key"] not in existing:
                pending.append(item)
                existing.add(item["memory_key"])
        self._append(pending)
        return len(pending)

    def append_iteration(self, record: Mapping[str, Any], *, source_run: str) -> bool:
        item = _iteration_record(source_run, record)
        if item["memory_key"] in self._keys():
            return False
        self._append((item,))
        return True

    def append_failure_event(self, event: Mapping[str, Any], *, source_run: str) -> bool:
        item = _failure_event(source_run, event)
        if item["memory_key"] in self._keys():
            return False
        self._append((item,))
        return True

    def records(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.archive_path)

    def planner_summary(self, *, recent_limit: int = 12, method_limit: int = 10) -> dict[str, Any]:
        """Return bounded evidence, with soft deprioritisation rather than bans."""
        records = self.records()
        completed = [item for item in records if item.get("record_type") == "iteration"]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in completed:
            grouped[_method_key(item)].append(item)
        method_evidence = []
        for method, items in grouped.items():
            scored = [item for item in items if isinstance(item.get("primary"), (int, float))]
            if not scored:
                continue
            deltas = [item["delta_primary"] for item in scored if isinstance(item.get("delta_primary"), (int, float))]
            best_delta = max(deltas, default=None)
            failures = sum(item.get("decision") in {"rejected", "failed"} for item in items)
            # This is advice, not a rule: the planner may revisit with a genuinely
            # different mechanism and must explain that distinction.
            status = "deprioritized" if len(scored) >= 2 and (best_delta is None or best_delta <= 0.0) else "observed"
            method_evidence.append({
                "method": method, "trials": len(items), "scored_trials": len(scored),
                "best_delta_primary": best_delta, "failed_trials": failures, "status": status,
            })
        method_evidence.sort(key=lambda item: (item["status"] != "deprioritized", -item["trials"], item["method"]))
        failures = [item for item in records if item.get("record_type") == "failure"]
        return {
            "record_count": len(records),
            "recent_evidence": completed[-recent_limit:],
            "method_evidence": method_evidence[:method_limit],
            "recent_implementation_failures": failures[-8:],
            "policy": "This is evidence, not a forbidden-method list. Prefer a materially new hypothesis. A revisit of a repeatedly weak method needs an explicit rationale describing what is different.",
        }

    def best_accepted_champion(self) -> dict[str, Any] | None:
        """Return the strongest reproducible accepted experiment from prior runs."""
        accepted = [
            record for record in self.records()
            if record.get("record_type") == "iteration"
            and record.get("decision") == "accepted"
            and isinstance(record.get("primary"), (int, float))
        ]
        if not accepted:
            return None
        champion = max(accepted, key=lambda item: float(item["primary"]))
        source_run = str(champion["source_run"])
        experiment_id = str(champion["experiment_id"])
        source_path = self.workspace_root / source_run / "patches" / f"{experiment_id}.patch"
        if not source_path.is_file():
            return None
        return {
            "source_run": source_run,
            "experiment_id": experiment_id,
            "primary": float(champion["primary"]),
            "hypothesis": str(champion.get("hypothesis", "")),
            "rationale": str(champion.get("rationale", "")),
            "config": dict(champion.get("config", {})),
            "source_path": str(source_path),
        }

    def _keys(self) -> set[str]:
        return {str(item.get("memory_key")) for item in self.records() if item.get("memory_key")}

    def _append(self, records: Iterable[Mapping[str, Any]]) -> None:
        items = list(records)
        if not items:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with self.archive_path.open("a", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(dict(item), sort_keys=True, default=str) + "\n")


def _iteration_record(source_run: str, record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    config = record.get("config") if isinstance(record.get("config"), Mapping) else {}
    experiment_id = str(record.get("experiment_id", "unknown"))
    return {
        "memory_key": f"iteration:{source_run}:{experiment_id}", "record_type": "iteration",
        "source_run": source_run, "experiment_id": experiment_id, "timestamp": record.get("timestamp"),
        "hypothesis": record.get("hypothesis", ""), "rationale": record.get("rationale", ""),
        "changed_factors": list(record.get("changed_factors", ())), "direction_id": record.get("direction_id", ""),
        "search_strategy": record.get("search_strategy", ""), "decision": record.get("decision", ""),
        "primary": metrics.get("primary"), "GAUC": metrics.get("GAUC"), "nDCG@5": metrics.get("nDCG@5"),
        "delta_primary": record.get("delta_primary"), "error": record.get("error"),
        "config": {key: value for key, value in config.items() if key != "_locked_settings"},
    }


def _failure_event(source_run: str, event: Mapping[str, Any]) -> dict[str, Any]:
    details = event.get("details") if isinstance(event.get("details"), Mapping) else {}
    experiment_id = str(event.get("experiment_id") or "run")
    action = str(event.get("action"))
    reason = details.get("reason") or details.get("error") or details.get("violations") or "unspecified failure"
    return {
        "memory_key": f"failure:{source_run}:{experiment_id}:{action}:{details.get('attempt_id', '')}:{reason}",
        "record_type": "failure", "source_run": source_run, "experiment_id": experiment_id,
        "timestamp": event.get("timestamp"), "action": action, "reason": str(reason),
        "failure_class": details.get("failure_class", action),
    }


def _method_key(item: Mapping[str, Any]) -> str:
    config = item.get("config") if isinstance(item.get("config"), Mapping) else {}
    return str(config.get("extension_name") or config.get("loss") or item.get("direction_id") or "unspecified")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records
