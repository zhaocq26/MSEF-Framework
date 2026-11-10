"""Deterministic engineering tools (quality-gate evaluation dimensions are produced automatically by deterministic tools).

All evaluation dimensions are output automatically by deterministic tools or computed from structured records, without relying on model judgment. This module provides the minimal toolset the quality gates depend on:

- Syntax check (AST / py_compile)
- Compile pass rate (module import check)
- Static analysis alerts (AST rules)
- Spec compliance check (naming/length rules)
- Test runner (pytest wrapper, injectable simulated results)
- Version control change statistics (git file list)
"""
from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Syntax check
# ---------------------------------------------------------------------------
def check_syntax(path: str) -> bool:
    """Syntax correctness: run AST parsing on a Python file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            ast.parse(f.read(), filename=path)
        return True
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False


def check_syntax_many(paths: Iterable[str]) -> Dict[str, bool]:
    return {p: check_syntax(p) for p in paths}


# ---------------------------------------------------------------------------
# Compile check
# ---------------------------------------------------------------------------
def check_compile(path: str) -> bool:
    """Compile pass rate: compile a single file (py_compile; falls back to compile() when sandboxed)."""
    import py_compile

    try:
        # Keep cfile in the source directory to avoid writing to system cache directories
        cfile = os.path.join(os.path.dirname(os.path.abspath(path)),
                             f".{os.path.basename(path)}c_check")
        try:
            py_compile.compile(path, cfile=cfile, doraise=True)
            return True
        except OSError:
            # Fall back to in-memory compilation when the directory is not writable
            with open(path, "r", encoding="utf-8") as f:
                compile(f.read(), path, "exec")
            return True
    except (OSError, py_compile.PyCompileError, SyntaxError, UnicodeDecodeError):
        return False


# ---------------------------------------------------------------------------
# Static analysis alerts (AST rules: null pointers / resource leaks / unhandled exceptions / type mismatches, etc.)
# ---------------------------------------------------------------------------
STATIC_RULES = {
    "bare_except": "bare except catches every exception",
    "unused_import": "unused import",
    "mutable_default": "mutable object used as a default argument",
    "empty_except": "empty except block",
    "print_statement": "leftover print debugging statement",
}


class AstStaticAnalyzer:
    """Lightweight AST-based static analyzer (outputs code quality alert counts)."""

    def analyze(self, code: str) -> List[Dict[str, str]]:
        alerts: List[Dict[str, str]] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return [{"rule": "syntax_error", "message": "syntax error"}]
        imported_names: set = set()
        used_names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_names.update(a.asname or a.name for a in node.names)
            elif isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Try):
                for h in node.handlers:
                    if h.type is None:
                        alerts.append({"rule": "bare_except", "message": STATIC_RULES["bare_except"]})
                        if not h.body or (len(h.body) == 1 and isinstance(h.body[0], ast.Pass)):
                            alerts.append({"rule": "empty_except", "message": STATIC_RULES["empty_except"]})
            elif isinstance(node, ast.FunctionDef):
                defaults = node.args.defaults + [d for d in node.args.kw_defaults if d]
                for d in defaults:
                    if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                        alerts.append({"rule": "mutable_default", "message": STATIC_RULES["mutable_default"]})
            elif isinstance(node, ast.Call):
                f = node.func
                if (isinstance(f, ast.Name) and f.id == "print"):
                    alerts.append({"rule": "print_statement", "message": STATIC_RULES["print_statement"]})
        for name in imported_names - used_names:
            alerts.append({"rule": "unused_import", "message": f"unused import: {name}"})
        return alerts


# ---------------------------------------------------------------------------
# Spec compliance check
# ---------------------------------------------------------------------------
SPEC_RULES = {
    "line_length": 120,
    "class_naming": True,
    "function_naming": True,
}


def check_spec_compliance(code: str, rules: Optional[Dict[str, Any]] = None) -> float:
    """Spec compliance: compliant code lines / total lines.

    Rules: line length limit (default 120), naming conventions (snake_case functions, CamelCase classes).
    """
    rules = rules or SPEC_RULES
    lines = code.splitlines()
    if not lines:
        return 1.0
    violations = 0
    max_len = int(rules.get("line_length", 120))
    class_re = re.compile(r"^\s*class\s+[a-z_]", re.M)
    func_re = re.compile(r"^\s*def\s+[A-Z]", re.M)
    for line in lines:
        if len(line) > max_len:
            violations += 1
    if rules.get("class_naming") and class_re.search(code):
        violations += 1
    if rules.get("function_naming") and func_re.search(code):
        violations += 1
    return max(0.0, 1.0 - violations / max(1, len(lines)))


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
@dataclass
class TestRunResult:
    """Test execution result."""

    passed: int = 0
    failed: int = 0
    skipped: int = 0
    total: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def unit_pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @classmethod
    def from_pytest(cls, exit_code: int, passed: int, failed: int,
                    skipped: int = 0, errors: Optional[List[str]] = None) -> "TestRunResult":
        return cls(passed=passed, failed=failed, skipped=skipped,
                   total=passed + failed + skipped, errors=errors or [])


def run_pytest(directory: str = ".", args: Optional[Sequence[str]] = None,
               timeout: int = 600) -> TestRunResult:
    """Run pytest and parse the results (integration test pass-rate data source)."""
    cmd = ["python", "-m", "pytest", "-q", "--no-header", str(directory)] + list(args or [])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout + proc.stderr
        passed = len(re.findall(r"^\.", out, re.M))
        failed = len(re.findall(r"^F", out, re.M))
        m = re.search(r"(\d+) passed", out)
        if m and not passed:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", out)
        if m and not failed:
            failed = int(m.group(1))
        return TestRunResult.from_pytest(proc.returncode, passed, failed)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return TestRunResult.from_pytest(-1, 0, 0, errors=[str(exc)])


# ---------------------------------------------------------------------------
# Version control
# ---------------------------------------------------------------------------
def git_changed_files(repo_path: str = ".", base: str = "HEAD") -> List[str]:
    """Changed file list (automatically counted by the version control tool: added/modified/deleted files)."""
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only", base, "--diff-filter=ACDMRT"],
            cwd=repo_path, capture_output=True, text=True, timeout=60,
        )
        if proc.returncode == 0:
            return [ln for ln in proc.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover
            logger.warning("git_changed_files failed: %s", exc)
    return []


# ---------------------------------------------------------------------------
# Impact scope reachability (reachability analysis from entry files along call and dependency graphs)
# ---------------------------------------------------------------------------
def impact_files_closure(entry_files: Iterable[str],
                         module_deps: Dict[str, List[str]]) -> set:
    """Compute the reachable closure of entry files over the module dependency graph (impact-scope identification data source)."""
    reachable: set = set()
    stack = list(entry_files)
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        for dep in module_deps.get(cur, []):
            if dep not in reachable:
                stack.append(dep)
    return reachable
