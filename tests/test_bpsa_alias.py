"""Tests for the bpsa namespace alias package."""


def test_import_bpsa():
    import bpsa

    assert bpsa is not None


def test_version_matches():
    import bpsa
    import smolagents

    assert bpsa.__version__ == smolagents.__version__


def test_top_level_import():
    from bpsa import CodeAgent
    from smolagents import CodeAgent as Original

    assert CodeAgent is Original


def test_submodule_agents():
    from bpsa.agents import CodeAgent
    from smolagents.agents import CodeAgent as Original

    assert CodeAgent is Original


def test_submodule_models():
    from bpsa.models import MessageRole
    from smolagents.models import MessageRole as Original

    assert MessageRole is Original


def test_submodule_tools():
    from bpsa.tools import Tool
    from smolagents.tools import Tool as Original

    assert Tool is Original


def test_sys_modules_registration():
    """After importing bpsa.agents, it should be registered in sys.modules."""
    import sys

    import bpsa.agents  # noqa: F401

    assert "bpsa.agents" in sys.modules
