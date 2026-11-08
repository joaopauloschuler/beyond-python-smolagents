import pytest

from smolagents.bp_utils import strip_trailing_tags


def test_removes_single_trailing_tag():
    code = "print('hi')</parameter>"
    assert strip_trailing_tags(code) == "print('hi')"


def test_removes_stacked_trailing_tags_and_whitespace():
    code = "x = 1\nprint(x)\n</parameter>\n</invoke>  \n"
    assert strip_trailing_tags(code) == "x = 1\nprint(x)"


def test_removes_opening_and_self_closing_tags():
    assert strip_trailing_tags("print(1)<br/>") == "print(1)"
    assert strip_trailing_tags("print(1)<foo>") == "print(1)"


def test_leaves_clean_code_untouched():
    code = "a = 1\nb = a + 1\nprint(b)"
    assert strip_trailing_tags(code) == code


def test_keeps_tags_inside_strings_and_mid_code():
    code = "print('</p>')\nhtml = '<div>'\nprint(html)"
    assert strip_trailing_tags(code) == code


def test_does_not_touch_comparisons():
    code = "ok = a < b > c"
    assert strip_trailing_tags(code) == code


def test_handles_empty_and_none():
    assert strip_trailing_tags("") == ""
    assert strip_trailing_tags(None) is None


def test_user_example():
    code = (
        "line371 = get_line_from_file('tasklist.md', 371)\n"
        "line372 = get_line_from_file('tasklist.md', 372)\n"
        "print(\"371:\", repr(line371))\n"
        "print(\"372:\", repr(line372))\n"
        "print(\"total lines:\", count_file_lines('tasklist.md'))</parameter>"
    )
    expected = code[: -len("</parameter>")]
    assert strip_trailing_tags(code) == expected


@pytest.mark.parametrize("tag_name", ["includefile", "include", "append"])
def test_keeps_trailing_include_tags(tag_name):
    one_tag = f"<{tag_name}>a.py</{tag_name}>"
    two_tags = f"{one_tag}\n<{tag_name}>b.py</{tag_name}>"
    assert strip_trailing_tags(one_tag) == one_tag
    assert strip_trailing_tags(two_tags) == two_tags
    assert strip_trailing_tags(f"{one_tag}  \n\t\n") == one_tag


@pytest.mark.parametrize("tag_name", ["includefile", "include", "append"])
def test_removes_stray_tags_after_trailing_include_tag(tag_name):
    one_tag = f"<{tag_name}>a.py</{tag_name}>"
    assert strip_trailing_tags(f"{one_tag}</parameter>") == one_tag
    assert strip_trailing_tags(f"{one_tag}\n</parameter>\n</invoke>\n") == one_tag
    assert strip_trailing_tags(f"{one_tag}</{tag_name}>") == one_tag


def test_removes_include_tag_halves():
    assert strip_trailing_tags("print(1)</includefile>") == "print(1)"
    assert strip_trailing_tags("print(1)\n<includefile>") == "print(1)"


def test_keeps_final_answer_with_as_text_tags():
    code = 'final_answer("""<includefile_as_text>a.py</includefile_as_text>""")'
    assert strip_trailing_tags(code) == code
