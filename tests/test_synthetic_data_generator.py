import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
GENERATOR_PATH = ROOT / "tools" / "generate_synthetic_data.py"
SPEC = importlib.util.spec_from_file_location("synthetic_data_generator", GENERATOR_PATH)
assert SPEC and SPEC.loader
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_generate_tasks_uses_expected_distribution_and_schema():
    records = generator.generate_tasks(50, seed=42)
    required_fields = {
        "task_id", "type", "title", "description", "avg_subtasks",
        "avg_changed_files", "affected_files", "required_risks", "complexity",
        "exact_match", "profile",
    }

    assert len(records) == 50
    assert Counter(record["type"] for record in records) == {
        "feature": 30, "bugfix": 15, "refactor": 5,
    }
    assert all(required_fields <= record.keys() for record in records)
    assert sum(record["exact_match"] for record in records) == 16
    assert {record["generation_seed"] for record in records} == {42}
    assert generator.generate_tasks(50, seed=42) == records


def test_generator_cli_writes_valid_json(tmp_path):
    output = tmp_path / "generated_tasks.json"
    subprocess.run(
        [
            sys.executable, str(GENERATOR_PATH), "--num_tasks", "10",
            "--output", str(output), "--seed", "9",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with output.open(encoding="utf-8") as handle:
        records = json.load(handle)
    assert Counter(record["type"] for record in records) == {
        "feature": 6, "bugfix": 3, "refactor": 1,
    }
