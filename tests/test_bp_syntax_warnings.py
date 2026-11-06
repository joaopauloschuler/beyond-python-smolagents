"""Every Python file under src/ must compile without warnings such as "invalid escape sequence"."""

import warnings
from pathlib import Path


SRC = Path(__file__).resolve().parent.parent / "src"


def test_src_compiles_without_syntax_warnings():
    # Python 3.12+ reports an invalid escape sequence as a SyntaxWarning (3.10 and 3.11: DeprecationWarning).
    # A future Python version will make it a SyntaxError, and the module would no longer import.
    files = sorted(SRC.rglob("*.py"))
    assert files, f"no Python files found under {SRC}"
    problems = []
    for path in files:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        problems += [
            f"{path.relative_to(SRC)}:{w.lineno}: {w.message}"
            for w in caught
            if issubclass(w.category, (SyntaxWarning, DeprecationWarning))
        ]
    assert problems == []
