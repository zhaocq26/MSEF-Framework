"""Verify no Chinese remains in Python comments or docstrings.

Comments (# ...) are detected via tokenize; docstrings via AST
(ast.get_docstring on Module/ClassDef/FunctionDef). Chinese inside
plain string literals (runtime data: labels, keywords, memory content,
prompt templates, demo output) is intentionally allowed and reported
separately for review.
"""
import ast
import io
import pathlib
import sys
import tokenize

CJK = lambda s: any("\u4e00" <= c <= "\u9fff" for c in s)


def scan(path):
    """Return (chinese_comments, chinese_docstrings, chinese_strings)."""
    src = path.read_text(encoding="utf-8")
    comments, strings = [], []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception as exc:  # pragma: no cover
        return [(f"TOKENIZE ERROR: {exc}",)], [], []
    for t in toks:
        if t.type == tokenize.COMMENT and CJK(t.string):
            comments.append((t.start[0], t.string.strip()))
        elif t.type == tokenize.STRING and CJK(t.string):
            strings.append((t.start[0], t.string[:60]))
    docstrings = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:  # pragma: no cover
        return comments + [(f"AST ERROR: {exc}",)], [], strings
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc and CJK(doc):
                lineno = getattr(node, "lineno", 1)
                docstrings.append((lineno, doc[:80]))
    return comments, docstrings, strings


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    targets = sorted(
        p
        for p in root.rglob("*.py")
        if ".venv" not in str(p) and ".pip-cache" not in str(p)
    )
    total_c = total_d = total_s = 0
    problems = []
    for p in targets:
        c, d, s = scan(p)
        if c and isinstance(c[0], tuple) and "ERROR" in str(c[0]):
            problems.append(f"{p}: {c[0][0]}")
            continue
        if c or d or s:
            print(
                f"{p.relative_to(root)}: comments={len(c)} docstrings={len(d)} strings={len(s)}"
            )
            total_c += len(c)
            total_d += len(d)
            total_s += len(s)
    print(
        f"\nTOTALS: chinese comments={total_c}, chinese docstrings={total_d}, "
        f"chinese strings={total_s}"
    )
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(" ", p)
        sys.exit(2)
    if total_c == 0 and total_d == 0:
        print("\nPASS: no Chinese in comments or docstrings.")
        sys.exit(0)
    print("\nFAIL: Chinese still present in comments/docstrings.")
    sys.exit(1)


if __name__ == "__main__":
    main()
