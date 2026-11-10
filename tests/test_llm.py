"""LLM backend tests (simulated backend + JSON parsing)."""
import pytest

from msef.llm.base import LLMBackend, parse_json_block
from msef.llm.simulated import SimulatedBackend


def test_simulated_backend_plan_output():
    """Simulated backend plan output is JSON with fields usable by the gates."""
    profile = {"plan": {"plan_completeness": 0.9, "impact_coverage": 0.8, "risk_sufficiency": 0.7}}
    backend = SimulatedBackend(profile=profile)
    raw = backend.complete("stage:plan planning a login API migration...")
    payload = parse_json_block(raw)
    assert payload["stage"] == "plan"
    assert payload["plan_completeness"] == 0.9
    assert payload["impact_coverage"] == 0.8
    assert "plan_elements" in payload


def test_simulated_backend_extracts_files_from_prompt():
    """Simulated backend extracts files from the prompt's subtask list (impact-scope data source)."""
    backend = SimulatedBackend(profile={})
    prompt = (
        "Task decomposition (subtasks):\n"
        "- [src/auth/login.py, src/auth/common.py] process file group\n"
        "- [tests/test_login.py] write tests\n"
        "Stage: plan planning..."
    )
    raw = backend.complete(prompt)
    payload = parse_json_block(raw)
    files = payload.get("files", [])
    assert "src/auth/login.py" in files
    assert "tests/test_login.py" in files


def test_simulated_backend_retry_boost():
    """After rollback suggestions are injected into the prompt, simulated output quality improves."""
    profile = {"code": {"spec_ok": 0.5}}
    backend = SimulatedBackend(profile=profile)
    plain = parse_json_block(backend.complete("stage:code coding implementation..."))
    retry = parse_json_block(backend.complete("stage:code coding implementation... (rollback suggestion: fix standards)"))
    assert retry["spec_ok"] > plain["spec_ok"]


def test_parse_json_block_with_fences():
    text = "```json\n{\"a\": 1, \"b\": [2, 3]}\n```"
    assert parse_json_block(text) == {"a": 1, "b": [2, 3]}


def test_parse_json_block_with_noise():
    text = "Result: {\"stage\": \"test\", \"unit_pass\": 0.9} complete"
    payload = parse_json_block(text)
    assert payload["stage"] == "test"
    assert payload["unit_pass"] == 0.9


def test_parse_json_block_raises_on_invalid():
    with pytest.raises(Exception):
        parse_json_block("output contains no JSON")
    with pytest.raises(Exception):
        parse_json_block("{\"invalid json")


def test_llm_token_count():
    backend = SimulatedBackend()
    assert backend.count_tokens("hello world") == 2
    assert backend.count_tokens("test") == 1
