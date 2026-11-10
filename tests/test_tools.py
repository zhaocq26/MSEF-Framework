"""Deterministic engineering tool tests (deterministic sources of quality-gate dimension data)."""
from msef.tools.static_tools import (
    AstStaticAnalyzer,
    check_compile,
    check_spec_compliance,
    check_syntax,
    impact_files_closure,
)


def test_check_syntax_valid_invalid(tmp_path):
    valid = tmp_path / "ok.py"
    valid.write_text("def f():\n    return 1\n")
    invalid = tmp_path / "bad.py"
    invalid.write_text("def f(:\n    return\n")
    assert check_syntax(str(valid)) is True
    assert check_syntax(str(invalid)) is False


def test_check_compile_valid(tmp_path):
    code = tmp_path / "mod.py"
    code.write_text("import math\n\ndef square(x):\n    return x * x\n")
    assert check_compile(str(code)) is True


def test_static_analyzer_detects_issues():
    analyzer = AstStaticAnalyzer()
    code = """
import os
import sys

def process(data=[]):
    try:
        result = data + [1]
        print(result)
    except:
        return None
    return result
"""
    alerts = analyzer.analyze(code)
    rules = {a["rule"] for a in alerts}
    assert "bare_except" in rules
    assert "unused_import" in rules          # os is unused
    assert "print_statement" in rules
    assert "mutable_default" in rules


def test_spec_compliance_line_length():
    long_line = "x = " + "a" * 200 + "\n" + "def ok():\n    pass\n"
    assert check_spec_compliance(long_line) < 1.0
    assert check_spec_compliance("def ok():\n    pass\n") == 1.0


def test_impact_files_closure():
    deps = {
        "a.py": ["b.py", "c.py"],
        "b.py": ["d.py"],
        "c.py": [],
        "d.py": ["e.py"],
    }
    reachable = impact_files_closure(["a.py"], deps)
    assert reachable == {"a.py", "b.py", "c.py", "d.py", "e.py"}
    # Unrelated files are unreachable
    assert "z.py" not in reachable