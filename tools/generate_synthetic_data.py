"""Generate reproducible enterprise software-engineering task corpora.

The output preserves the task contract used by ``examples/run_demo.py`` and
adds type-level workload metadata for dataset analysis.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


TASK_PROFILES = {
    "feature": {
        "prefix": "F",
        "share": 0.60,
        "avg_subtasks": 8.3,
        "avg_changed_files": 12.7,
        "label": "Feature Development",
    },
    "bugfix": {
        "prefix": "B",
        "share": 0.30,
        "avg_subtasks": 4.1,
        "avg_changed_files": 5.2,
        "label": "Bug Fixing",
    },
    "refactor": {
        "prefix": "R",
        "share": 0.10,
        "avg_subtasks": 6.7,
        "avg_changed_files": 18.4,
        "label": "Refactoring",
    },
}

SERVICE_CATALOG = [
    ("user login", "identity", "identity"),
    ("payment gateway", "payments", "payments"),
    ("order tracking", "orders", "orders"),
    ("inventory reservation", "inventory", "inventory"),
    ("tenant billing", "billing", "billing"),
    ("notification delivery", "notifications", "notifications"),
    ("customer support case routing", "support", "support"),
    ("audit event export", "audit", "audit"),
]

FEATURE_ACTIONS = [
    "add a Redis-backed cache",
    "implement request validation",
    "introduce idempotency keys",
    "add asynchronous retry handling",
    "expose a versioned REST endpoint",
    "add tenant-scoped rate limiting",
]

BUGFIX_INCIDENTS = [
    "duplicate processing after retry",
    "connection leakage under load",
    "stale reads after an asynchronous update",
    "incorrect timezone handling at a reporting boundary",
    "missing authorization enforcement on a fallback path",
    "lost events during consumer rebalancing",
]

REFACTOR_ACTIONS = [
    "extract the integration boundary into a gRPC service",
    "replace untyped payloads with versioned schemas",
    "split the monolithic workflow into independently deployable modules",
    "introduce a provider-neutral adapter interface",
    "consolidate duplicated configuration loading behind typed settings",
    "separate read and write paths behind explicit interfaces",
]

RISK_LIBRARY = {
    "feature": [
        "backward compatibility", "tenant isolation", "retry duplication",
        "latency regression", "authorization bypass", "rollback readiness",
    ],
    "bugfix": [
        "regression coverage", "data consistency", "retry behavior",
        "customer impact during rollout", "observability gaps", "rollback safety",
    ],
    "refactor": [
        "consumer compatibility", "schema migration", "service discovery",
        "deployment sequencing", "performance regression", "rollback compatibility",
    ],
}


def allocate_task_counts(num_tasks: int) -> dict[str, int]:
    """Allocate exactly ``num_tasks`` using the largest-remainder method."""
    if num_tasks < 1:
        raise ValueError("num_tasks must be at least 1")

    raw_counts = {
        task_type: num_tasks * profile["share"]
        for task_type, profile in TASK_PROFILES.items()
    }
    counts = {task_type: math.floor(value) for task_type, value in raw_counts.items()}
    remainder = num_tasks - sum(counts.values())
    ranked_types = sorted(
        TASK_PROFILES,
        key=lambda task_type: (raw_counts[task_type] - counts[task_type], TASK_PROFILES[task_type]["share"]),
        reverse=True,
    )
    for task_type in ranked_types[:remainder]:
        counts[task_type] += 1
    return counts


def _file_candidates(service: str, module: str, task_type: str) -> list[str]:
    candidates = [
        f"services/{service}/api.py",
        f"services/{service}/handler.py",
        f"services/{service}/service.py",
        f"services/{service}/repository.py",
        f"services/{service}/models.py",
        f"services/{service}/settings.py",
        f"services/{service}/metrics.py",
        f"services/{service}/events.py",
        f"services/{service}/worker.py",
        f"api/{module}/routes.py",
        f"api/{module}/schemas.py",
        f"clients/{service}_client.py",
        f"tests/{service}/test_api.py",
        f"tests/{service}/test_service.py",
        f"tests/{service}/test_repository.py",
        f"tests/integration/test_{service}_workflow.py",
        f"deploy/{service}/deployment.yaml",
        f"docs/{service}/runbook.md",
    ]
    if task_type == "refactor":
        candidates.extend([
            f"services/{service}/contracts.py",
            f"services/{service}/adapters.py",
            f"services/{service}/migration.py",
            f"events/{service}_schema.py",
            f"events/{service}_codec.py",
            f"tests/contract/test_{service}_compatibility.py",
        ])
    return candidates


def _sample_file_count(task_type: str, rng: random.Random) -> int:
    target = TASK_PROFILES[task_type]["avg_changed_files"]
    spread = max(1.0, target * 0.18)
    return max(1, round(rng.gauss(target, spread)))


def _select_files(service: str, module: str, task_type: str, count: int, rng: random.Random) -> list[str]:
    candidates = _file_candidates(service, module, task_type)
    rng.shuffle(candidates)
    selected = candidates[:min(count, len(candidates))]
    while len(selected) < count:
        sequence = len(selected) - len(candidates) + 1
        selected.append(f"services/{service}/internal/change_{sequence:02d}.py")
    return sorted(selected)


def _make_text(task_type: str, domain: str, service: str, rng: random.Random) -> tuple[str, str]:
    title_domain = domain.title()
    if task_type == "feature":
        action = rng.choice(FEATURE_ACTIONS)
        title = f"{title_domain} capability expansion"
        description = (
            f"Implement {action} for the {domain} service while preserving tenant isolation, "
            "operational observability, and a documented rollback path."
        )
    elif task_type == "bugfix":
        incident = rng.choice(BUGFIX_INCIDENTS)
        title = f"{title_domain} reliability correction"
        description = (
            f"Fix {incident} in the {domain} service and add regression coverage for concurrent "
            "requests, retry paths, and production-safe rollback."
        )
    else:
        action = rng.choice(REFACTOR_ACTIONS)
        title = f"{title_domain} boundary refactoring"
        description = (
            f"Refactor the {domain} service to {action}, preserving public contracts, "
            "observability, and an incremental migration path."
        )
    return title, description


def _make_record(task_type: str, index: int, rng: random.Random) -> dict[str, Any]:
    profile = TASK_PROFILES[task_type]
    domain, service, module = rng.choice(SERVICE_CATALOG)
    title, description = _make_text(task_type, domain, service, rng)
    changed_file_count = _sample_file_count(task_type, rng)
    return {
        "task_id": f"{profile['prefix']}{index:03d}",
        "type": task_type,
        "title": title,
        "description": description,
        "avg_subtasks": profile["avg_subtasks"],
        "avg_changed_files": profile["avg_changed_files"],
        "affected_files": _select_files(service, module, task_type, changed_file_count, rng),
        "required_risks": sorted(rng.sample(RISK_LIBRARY[task_type], k=3)),
        "complexity": round(rng.uniform(0.48, 0.86) if task_type != "refactor" else rng.uniform(0.64, 0.96), 2),
        "exact_match": 0,
        "profile": rng.choices(["good", "medium", "hard"], weights=[2, 3, 4], k=1)[0],
    }


def generate_tasks(num_tasks: int, seed: int = 42) -> list[dict[str, Any]]:
    """Create a seeded corpus with the configured type distribution."""
    rng = random.Random(seed)
    counts = allocate_task_counts(num_tasks)
    records: list[dict[str, Any]] = []
    for task_type in ("feature", "bugfix", "refactor"):
        records.extend(
            _make_record(task_type, index, rng)
            for index in range(1, counts[task_type] + 1)
        )
    rng.shuffle(records)
    exact_match_count = round(num_tasks * 0.33)
    for index in rng.sample(range(num_tasks), k=exact_match_count):
        records[index]["exact_match"] = 1
    for record in records:
        record["generation_seed"] = seed
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic MSEF task corpus.")
    parser.add_argument("--num-tasks", "--num_tasks", dest="num_tasks", type=int, default=50,
                        help="Number of tasks to generate (default: 50).")
    parser.add_argument("--output", type=Path, default=Path("data/generated_tasks.json"),
                        help="Output JSON path (default: data/generated_tasks.json).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible corpora (default: 42).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        records = generate_tasks(args.num_tasks, args.seed)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")

    counts = allocate_task_counts(args.num_tasks)
    print(f"Wrote {len(records)} tasks to {args.output}")
    print(f"Seed: {args.seed}")
    print("Distribution: " + ", ".join(f"{task_type}={counts[task_type]}" for task_type in TASK_PROFILES))


if __name__ == "__main__":
    main()
