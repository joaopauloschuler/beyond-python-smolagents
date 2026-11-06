"""Tests for the agent's parse_tags(), which finds the tags that save or append files in the model's output.

parse_tags() works with any tag name, so these tests use a neutral one.
"""

import pytest

from smolagents.agents import CodeAgent
from smolagents.models import Model


@pytest.fixture(scope="module")
def agent():
    code_agent = CodeAgent(tools=[], model=Model())
    yield code_agent
    code_agent.cleanup()


@pytest.mark.parametrize("quote", ['"', "'"], ids=["double-quotes", "single-quotes"])
def test_reads_the_filename_in_either_quote(agent, quote):
    text = f"<note filename={quote}a.txt{quote}>hello</note>"
    assert agent.parse_tags("note", text) == [{"filename": "a.txt", "content": "hello"}]


def test_keeps_multi_line_content_and_finds_every_tag(agent):
    text = '<note filename="a.py">line 1\nline 2\n</note> text <note\n  filename="b.txt">x</note>'
    assert agent.parse_tags("note", text) == [
        {"filename": "a.py", "content": "line 1\nline 2\n"},
        {"filename": "b.txt", "content": "x"},
    ]


def test_ignores_mismatched_quotes_and_other_tags(agent):
    text = '<note filename="a.txt\'>x</note> <other filename="b.txt">y</other>'
    assert agent.parse_tags("note", text) == []
