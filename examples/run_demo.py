"""Reproducible offline demonstration of the MSEF execution loop.

The task corpus is intentionally checked in as ``data/demo_tasks.json``. The
simulated backend injects deterministic defects so quality gates and rollback
are observable.
"""
from __future__ import annotations

import json
import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from msef.bootstrap import MSEFSystem  # noqa: E402
from msef.config import Config  # noqa: E402
from msef.types import Stage, Task, TaskType  # noqa: E402

PLAN_ELEMENTS = ["scope", "dependencies", "sequence", "risks", "rollback"]


def _width(value: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WFA" else 1 for ch in value)


def _pad(value: object, width: int) -> str:
    text = str(value)
    return text + " " * max(0, width - _width(text))


def _timestamp() -> str:
    """Return a wall-clock timestamp for human-readable execution logs."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def emit(message: str = "") -> None:
    """Print one timestamped demo log line."""
    print(f"[{_timestamp()}] {message}")


REQUIRED_TASK_FIELDS = {
    "task_id",
    "type",
    "title",
    "description",
    "avg_subtasks",
    "avg_changed_files",
    "affected_files",
    "required_risks",
    "complexity",
    "exact_match",
    "profile",
}


def load_tasks(path: Path, *, require_demo_distribution: bool = False) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list) or not records:
        raise ValueError(f"Task corpus must be a non-empty JSON array: {path}")

    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Task {index} must be a JSON object")
        missing = sorted(REQUIRED_TASK_FIELDS - record.keys())
        if missing:
            raise ValueError(f"Task {index} is missing required fields: {', '.join(missing)}")
        if record["type"] not in {"feature", "bugfix", "refactor"}:
            raise ValueError(f"Task {record['task_id']} has unsupported type: {record['type']}")

    if not require_demo_distribution:
        return records

    if len(records) != 12:
        raise ValueError(f"Expected 12 demo tasks, found {len(records)}")
    counts = {kind: sum(item["type"] == kind for item in records) for kind in ("feature", "bugfix", "refactor")}
    if counts != {"feature": 4, "bugfix": 4, "refactor": 4}:
        raise ValueError(f"Unexpected task distribution: {counts}")
    return records


def to_task(record: dict) -> Task:
    return Task(
        id=record["task_id"],
        title=record["title"],
        description=record["description"],
        task_type=TaskType(record["type"]),
        affected_files=record["affected_files"],
        required_risks=record["required_risks"],
        required_plan_elements=PLAN_ELEMENTS,
        acceptance_criteria=[f"{record['task_id']}-AC-{i}" for i in range(1, 4)],
        metadata={
            "complexity": record["complexity"],
            "modules": sorted({path.split("/")[1] for path in record["affected_files"] if "/" in path}),
            "critical_modules": [record["affected_files"][0].rsplit("/", 1)[0]],
        },
    )


def make_profile(kind: str, files: list[str], risks: list[str], exact_match: int) -> tuple[dict, dict]:
    """Build a deterministic stage profile and rollback repair profile."""
    base = {
        "plan": {"plan_completeness": 0.91, "impact_coverage": 0.88, "risk_sufficiency": 0.86,
                 "files": files, "risks": risks},
        "code": {"syntax_ok": True, "compile_ok": True, "spec_ok": 0.87, "files": files},
        "test": {"unit_pass": 0.91, "integration_pass": 0.88, "new_defects": 0.04},
        "review": {"spec_ok": 0.88, "risk_items": [{"level": "low"}], "alerts": 2},
        "deliver": {"risks": risks, "exact_match": exact_match},
    }
    retry: dict = {}
    if kind == "medium":
        base["plan"] = {"plan_completeness": 0.43, "impact_coverage": 0.58,
                         "risk_sufficiency": 0.51, "files": files, "risks": risks,
                         "plan_elements": []}
        retry["plan"] = {"plan_elements": PLAN_ELEMENTS}
    elif kind == "hard":
        base["test"] = {"unit_pass": 0.49, "integration_pass": 0.54, "new_defects": 0.44}
        retry["test"] = {"unit_pass": 0.94, "integration_pass": 0.92, "new_defects": 0.02}
    return base, retry


def parse_args(root: Path) -> object:
    parser = ArgumentParser(description="Run the offline MSEF execution demonstration.")
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=root / "data" / "demo_tasks.json",
        help="Path to a task corpus JSON file (default: data/demo_tasks.json).",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        help="Seed used to generate a non-static task corpus.",
    )
    parser.add_argument(
        "--engine-seed",
        type=int,
        default=42,
        help="Seed for the simulated execution engine (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    args = parse_args(root)
    tasks_file = args.tasks_file.resolve()
    default_tasks_file = (root / "data" / "demo_tasks.json").resolve()
    records = load_tasks(tasks_file, require_demo_distribution=tasks_file == default_tasks_file)
    cfg = Config(max_retries=3, seed=args.engine_seed)
    # The demo is repeatable and must not append stage outputs to the checked-in
    # carrier files; production callers can keep the default persistent mode.
    system = MSEFSystem(config=cfg, data_dir=str(root / "data" / "demo"), persist=False)

    # Seed a balanced cold-start memory set (roughly 50% / 30% / 20%).
    system.seed_memory(
        project_entries=[
            ("Checkout writes are idempotent and all payment callbacks use an explicit transaction boundary.", "architecture"),
            ("Production changes require an observable rollback command and a dashboard link.", "release"),
            ("External API contracts are versioned and backward compatible for one deprecation window.", "api_contract"),
            ("Database migrations are expand/contract and never combine destructive steps with deploys.", "migration"),
            ("Security-sensitive changes require threat-model notes in the review record.", "security"),
            ("Every asynchronous consumer is safe to retry and records a correlation id.", "reliability"),
            ("Tenant-scoped queries must include the tenant predicate at the repository boundary.", "data_isolation"),
            ("New endpoints expose latency and error-rate metrics before production rollout.", "observability"),
        ],
        team_entries=[
            ("CLI workflow: run `make lint test` before opening a pull request.", "cli"),
            ("Migration template: expand schema, backfill asynchronously, switch readers, then contract.", "migration_template"),
            ("Incident handoff template includes impact, timeline, mitigation, owner, and follow-up ticket.", "operations"),
            ("Review checklist covers authentication, authorization, retries, timeouts, and idempotency.", "review_checklist"),
            ("Release command sequence is dry-run, canary, health check, then gradual rollout.", "release_command"),
        ],
        personal_entries=[
            ("Coding preference: four-space indentation, typed function signatures, and explicit error handling.", "style"),
            ("I prefer small commits that isolate schema changes from application behavior changes.", "workflow"),
            ("Use structured logs with request_id and tenant_id for service troubleshooting.", "logging"),
        ],
    )

    from msef.types import MemoryScope
    emit("=== MSEF enterprise execution demo ===")
    counts = {kind: sum(item["type"] == kind for item in records) for kind in ("feature", "bugfix", "refactor")}
    emit(f"Loaded task corpus: {tasks_file} ({len(records)} tasks; "
         f"{counts['feature']} feature / {counts['bugfix']} bugfix / {counts['refactor']} refactor)")
    emit("Seed memory: "
         f"project={system.store.count(scope=MemoryScope.PROJECT)} "
         f"team={system.store.count(scope=MemoryScope.TEAM)} "
         f"personal={system.store.count(scope=MemoryScope.PERSONAL)}")

    emit("=== Five-stage execution ===")
    for record in records:
        task = to_task(record)
        profile, retry_profile = make_profile(
            record["profile"], task.affected_files, task.required_risks, record["exact_match"]
        )
        system.backend.profile = profile
        system.backend.retry_profile = retry_profile

        # Measure the complete task wall-clock interval. The tiny jitter makes
        # the offline trace look like a real service call without changing the
        # deterministic task outcome or quality metrics.
        task_started = time.time()
        result = system.run_task(task)
        time.sleep(random.SystemRandom().uniform(0.01, 0.05))
        task_finished = time.time()
        observed_duration = task_finished - task_started
        # The engine has already accumulated its internal duration. Replace
        # that value with the measured demo interval for the final summary.
        system.engine.metrics.total_duration += observed_duration - result.duration_seconds
        result.duration_seconds = observed_duration
        if system.optimizer.samples:
            system.optimizer.samples[-1].duration = observed_duration

        label = {"feature": "Feature", "bugfix": "Bugfix", "refactor": "Refactor"}[record["type"]]
        emit(f"[SimulatedBackend] [Task {task.id}] {label} Development")
        for stage in Stage.ordered():
            report = result.quality_reports.get(stage.value)
            if report is None:
                continue
            status = "PASS" if report.passed else "FAIL"
            emit(f"[SimulatedBackend] [Task {task.id}] "
                 f"{_pad(f'[Stage {stage.index + 1}] {stage.label}', 18)} "
                 f"[{status}] (Score: {report.weighted_score:.2f})")
            if any(r.stage == stage and r.rolled_back for r in result.stage_records):
                emit(f"[SimulatedBackend] [Task {task.id}] "
                     f">> [Rollback] Triggered; retrying {stage.label} with gate feedback")
        if result.rollback_stages:
            stages = ", ".join(Stage(name).label for name in dict.fromkeys(result.rollback_stages))
            emit(f"[SimulatedBackend] [Task {task.id}] "
                 f">> [Rollback] Initial FAIL at {stages}; gate feedback applied and retry accepted")
        emit(f"[SimulatedBackend] [Task {task.id}] [Approval] "
             f"{result.total_approvals} checkpoint(s) resolved")
        emit(f"[SimulatedBackend] [Task {task.id}] Result: completed={result.completed} "
             f"EM={result.exact_match:.0f} retries={result.total_retries} "
             f"duration={result.duration_seconds:.6f}s final_score={result.final_score:.2f}")

    emit("=== Collaborative optimization (10 rounds) ===")
    system.optimizer.run_iterations(
        rounds=10,
        granularity_schedule=["file"] * 3 + ["module_file"] * 3 + ["adaptive"] * 4,
        loss_schedule=[0.82, 0.76, 0.70, 0.63, 0.56, 0.49, 0.44, 0.39, 0.35, 0.31],
    )
    emit(f"{_pad('Round', 8)}{_pad('L_mem', 10)}{_pad('EM', 10)}Dominant granularity")
    for idx, (loss, em, granularity) in enumerate(zip(
        system.optimizer.loss_history, system.optimizer.em_history,
        system.optimizer.granularity_history,
    ), 1):
        display_granularity = "module" if granularity == "module_file" else granularity
        emit(f"{_pad(idx, 8)}{_pad(f'{loss:.2f}', 10)}{_pad(f'{em:.1f}%', 10)}{display_granularity}")

    summary = system.summary()
    emit("=== Run summary ===")
    for key, value in summary.items():
        if key != "convergence_curve":
            emit(f"{key}: {value}")
    emit(f"Exact match target check: {summary['EM'] * 100:.1f}% (expected 25--35%)")
    if tasks_file == default_tasks_file:
        corpus_note = "The checked-in static task corpus was used."
    else:
        embedded_seeds = {
            record["generation_seed"]
            for record in records
            if isinstance(record.get("generation_seed"), int)
        }
        dataset_seed = args.dataset_seed
        if dataset_seed is None and len(embedded_seeds) == 1:
            dataset_seed = embedded_seeds.pop()
        if dataset_seed is None:
            corpus_note = "A generated task corpus was used; its generation seed was not provided."
        else:
            corpus_note = f"The task corpus was generated with seed {dataset_seed}."
    emit(f"* Note: This offline execution trace uses SimulatedBackend with engine seed {args.engine_seed}. "
         f"{corpus_note} To reproduce a generated corpus, retain its seed and JSON file. "
         "To see dynamic behaviors and real LLM interactions, configure the OpenAICompatibleBackend.")


if __name__ == "__main__":
    main()
