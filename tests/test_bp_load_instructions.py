"""Tests for load_agent_instructions() and README fallback."""

import os
import tempfile

import pytest


def test_no_files_returns_none():
    """When no instruction files or READMEs exist, return None."""
    from smolagents.bp_cli import load_agent_instructions

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            result = load_agent_instructions()
            assert result is None
        finally:
            os.chdir(old_cwd)


def test_instruction_file_loaded():
    """When CLAUDE.md exists, it should be loaded (no README fallback)."""
    from smolagents.bp_cli import load_agent_instructions

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            with open(os.path.join(tmpdir, "CLAUDE.md"), "w") as f:
                f.write("Some instructions here")
            result = load_agent_instructions()
            assert result is not None
            assert "Some instructions here" in result
            assert "<filenotes>" in result
        finally:
            os.chdir(old_cwd)


def test_instruction_file_takes_priority_over_readme():
    """When both CLAUDE.md and README.md exist, only CLAUDE.md is loaded."""
    from smolagents.bp_cli import load_agent_instructions

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            with open(os.path.join(tmpdir, "CLAUDE.md"), "w") as f:
                f.write("Instructions")
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write("Project readme")
            result = load_agent_instructions()
            assert "<filenotes>" in result
            assert "<readme>" not in result
            assert "Project readme" not in result
        finally:
            os.chdir(old_cwd)


def test_readme_md_fallback():
    """When no instruction files exist but README.md does, load it."""
    from smolagents.bp_cli import load_agent_instructions

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write("# My Project\nThis is a project.")
            result = load_agent_instructions()
            assert result is not None
            assert "<readme>" in result
            assert "# My Project" in result
            assert "general context" in result
        finally:
            os.chdir(old_cwd)


def test_readme_case_insensitive():
    """README matching should be case-insensitive."""
    from smolagents.bp_cli import load_agent_instructions

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            with open(os.path.join(tmpdir, "Readme.md"), "w") as f:
                f.write("Mixed case readme")
            result = load_agent_instructions()
            assert result is not None
            assert "Mixed case readme" in result
            assert "<readme>" in result
        finally:
            os.chdir(old_cwd)


def test_readme_txt_fallback():
    """readme.txt should be loaded as fallback."""
    from smolagents.bp_cli import load_agent_instructions

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            with open(os.path.join(tmpdir, "readme.txt"), "w") as f:
                f.write("Text readme")
            result = load_agent_instructions()
            assert result is not None
            assert "Text readme" in result
        finally:
            os.chdir(old_cwd)


def test_readme_md_preferred_over_txt():
    """README.md should be preferred over readme.txt."""
    from smolagents.bp_cli import load_agent_instructions

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write("Markdown readme")
            with open(os.path.join(tmpdir, "readme.txt"), "w") as f:
                f.write("Text readme")
            result = load_agent_instructions()
            assert result is not None
            assert "Markdown readme" in result
            assert "Text readme" not in result
        finally:
            os.chdir(old_cwd)


def test_readme_no_extension():
    """A file named just 'README' (no extension) should be loaded."""
    from smolagents.bp_cli import load_agent_instructions

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            with open(os.path.join(tmpdir, "README"), "w") as f:
                f.write("Plain readme")
            result = load_agent_instructions()
            assert result is not None
            assert "Plain readme" in result
        finally:
            os.chdir(old_cwd)


def test_readme_truncation():
    """READMEs longer than README_MAX_CHARS should be truncated."""
    from smolagents.bp_cli import README_MAX_CHARS, load_agent_instructions

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            long_content = "A" * (README_MAX_CHARS + 5000)
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write(long_content)
            result = load_agent_instructions()
            assert result is not None
            # The content inside should be truncated
            assert "A" * README_MAX_CHARS in result
            assert "A" * (README_MAX_CHARS + 1) not in result
        finally:
            os.chdir(old_cwd)


def test_empty_readme_ignored():
    """An empty README file should not be loaded."""
    from smolagents.bp_cli import load_agent_instructions

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write("")
            result = load_agent_instructions()
            assert result is None
        finally:
            os.chdir(old_cwd)


def test_readme_rst_fallback():
    """README.rst should be loaded as fallback."""
    from smolagents.bp_cli import load_agent_instructions

    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            with open(os.path.join(tmpdir, "README.rst"), "w") as f:
                f.write("RST readme content")
            result = load_agent_instructions()
            assert result is not None
            assert "RST readme content" in result
        finally:
            os.chdir(old_cwd)
