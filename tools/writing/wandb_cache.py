import json
import os
from dataclasses import dataclass


@dataclass
class CachedRun:
    config: dict
    summary: dict
    tags: list
    name: str
    id: str


def _jsonify(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return str(value)


def _serialize_runs(runs):
    return [
        {
            "config": _jsonify(dict(run.config)),
            "summary": _jsonify(dict(run.summary)),
            "tags": _jsonify(list(run.tags or [])),
            "name": run.name or "",
            "id": run.id or "",
        }
        for run in runs
    ]


def _deserialize_runs(records):
    return [
        CachedRun(
            config=record.get("config", {}),
            summary=record.get("summary", {}),
            tags=record.get("tags", []),
            name=record.get("name", ""),
            id=record.get("id", ""),
        )
        for record in records
    ]


def save_cached_runs(cache_path: str, baselines, ours_complete, ours_empty):
    payload = {
        "baselines": _serialize_runs(baselines),
        "ours_complete": _serialize_runs(ours_complete),
        "ours_empty": _serialize_runs(ours_empty),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_cached_runs(cache_path: str):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return (
        _deserialize_runs(payload.get("baselines", [])),
        _deserialize_runs(payload.get("ours_complete", [])),
        _deserialize_runs(payload.get("ours_empty", [])),
    )
