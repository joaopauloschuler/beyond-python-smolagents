# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

import ast
import glob
import os
import re
import shutil
import sys
import time
from textwrap import dedent


# Private-use characters stand for <runcode> and </runcode> while parsing, so the model's markdown cannot match them.
CODE_BLOCK_OPEN = "\ue000"
CODE_BLOCK_CLOSE = "\ue001"
# Stands for "<" in text whose tags are not tags, so that whole-text tag matching cannot see them.
HIDDEN_TAG_BRACKET = "\ue002"
MARKDOWN_FENCE_AROUND_CODE = re.compile(r"\A```(?:python|py)\b\s*(.*?)\s*```\Z", re.DOTALL)
FINAL_ANSWER_INCLUDE_TAG = re.compile(r"<(includefile|include|append)>(.*?)</\1>", re.DOTALL)
INCLUDE_AS_TEXT_TAG = re.compile(r"<(includefile|include|append)_as_text>(.*?)</\1_as_text>", re.DOTALL)
SECTION_OPENING = r"""(?i:<(?P<section>savetofile|appendtofile|thoughts|plans|freewill|observations)(?:\s+filename=(?P<quote>['"])[^'"]*(?P=quote))?>)"""
# What the top level of a model output can hold: skipped sections (matching parse_tags), code blocks and bare stop markers.
OUTSIDE_BLOCK_TAG = re.compile(
    SECTION_OPENING +
    r"|<(?P<code_block>runcode|code|final_answer)>"
    r"|(?P<stray_stop></runcode>|</code>|</final_answer>|Calling tools:)"
)
PYTHON_STRING_OR_STOP = re.compile(
    r"""\#(?:(?!</runcode>|</code>|</final_answer>)[^\n])*"""
    r"""|(?<!\\)\"\"\"[^"\\]*(?:(?:\\.|"(?!""))[^"\\]*)*\"\"\""""
    r"""|(?<!\\)\'\'\'[^'\\]*(?:(?:\\.|'(?!''))[^'\\]*)*\'\'\'"""
    r"""|"[^"\\\n]*(?:\\.[^"\\\n]*)*\""""
    r"""|'[^'\\\n]*(?:\\.[^'\\\n]*)*'"""
    r"""|(?P<stop></runcode>|</code>|</final_answer>|Calling tools:)""",
    re.DOTALL,
)
SECTION_OPENING_OR_PYTHON_STRING = re.compile(SECTION_OPENING + "|" + PYTHON_STRING_OR_STOP.pattern, re.DOTALL)
TRIPLE_QUOTED_STRING_ESCAPES = str.maketrans({"\\": "\\\\", '"': '\\"', "\r": "\\r", "\0": "\\x00"})

dotenv_loaded_names: set[str] = set()


def load_dotenv_recording_names(env_path: str | None = None) -> set[str]:
    """load_dotenv that adds the names it set to dotenv_loaded_names and returns them; without env_path it searches
    as load_dotenv does. All loads go through here because a name loaded earlier looks like a process env var later."""
    from dotenv import dotenv_values, find_dotenv, load_dotenv

    env_path = env_path or find_dotenv()
    if not env_path:
        return set()
    added_names = {k for k, v in dotenv_values(env_path).items() if v is not None and k not in os.environ}
    load_dotenv(env_path)
    dotenv_loaded_names.update(added_names)
    return added_names


ENV_TRUE_VALUES = ("1", "true", "yes", "on")
ENV_FALSE_VALUES = ("0", "false", "no", "off", "")
_warned_env_values: set[tuple[str, str]] = set()


def parse_env_bool(name: str) -> bool | None:
    """Env var `name` as a bool, ignoring case and outer spaces; None when it is unset or invalid.
    An invalid value prints one warning to stderr per name and value, because callers read some settings every step."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    value = raw_value.strip().lower()
    if value in ENV_TRUE_VALUES:
        return True
    if value in ENV_FALSE_VALUES:
        return False
    if (name, raw_value) not in _warned_env_values:
        _warned_env_values.add((name, raw_value))
        print(f"Ignoring {name}={raw_value!r}: expected 1, true, yes, on, 0, false, no or off.", file=sys.stderr)
    return None


def get_env_bool(name: str, default: bool = False) -> bool:
    """Env var `name` as a bool (see parse_env_bool); default when it is unset or invalid."""
    value = parse_env_bool(name)
    return default if value is None else value

def get_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)

def get_env_int(name: str, default: int) -> int:
    val = get_env(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        console.print(f"[yellow]Warning: Invalid integer for {name}='{val}', using default: {default}[/]")
        return default


def delay_execution_10(pagent, **kwargs) -> bool:
    """
    Delays the execution for 10 seconds.
    """
    time.sleep(10)
    return True

def delay_execution_30(pagent, **kwargs) -> bool:
    """
    Delays the execution for 30 seconds.
    """
    time.sleep(30)
    return True

def delay_execution_120(pagent, **kwargs) -> bool:
    """
    Delays the execution for 120 seconds.
    """
    time.sleep(120)
    return True

def remove_folder_contents(folder_name):
  if os.path.exists(folder_name):
    for item in os.listdir(folder_name):
      item_path = os.path.join(folder_name, item)
      if os.path.isfile(item_path):
        os.remove(item_path)
      elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

def copy_folder_contents(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dest_path = os.path.join(dest_folder, item)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path)

def remove_files(file_filter):
  """Removes all files in file_filter."""
  txt_files = glob.glob(file_filter)
  for file_path in txt_files:
    os.remove(file_path)

def is_valid_python_code(code_string):
    """Returns true if the string is a valid python code."""
    result = False
    try:
        ast.parse(code_string)
        return True
    except SyntaxError:
        result = False
    except Exception:
        result = False
    return result

def fix_nested_tags(tagname, text):
    """
    Replaces outer <tagname> opening tags with "tagname" text,
    keeping only the innermost <tagname></tagname> pair as actual tags.
    
    Rule: If you find 2 opening tags <tagname> without a closing tag in the middle,
    the first tag should be replaced by "tagname".
    """
    result = text
    tagname_len = len(tagname)

    # Keep replacing until no more replacements are needed
    while True:
        # Look for pattern: <run> followed eventually by another <run>
        # with no </run> in between
        pattern = r'<'+tagname+r'>((?:(?!</'+tagname+r').)*?)<'+tagname+r'>'
        match = re.search(pattern, result, re.DOTALL)

        if not match:
            break

        # Replace the FIRST <run> with "run"
        start_pos = match.start()
        end_of_first_tag = start_pos + tagname_len + 2  # Length of '<run>'

        result = result[:start_pos] + '"'+tagname+'"' + result[end_of_first_tag:]

    return result

def end_of_block(text, start, block):
    """Returns where the block opened before start ends: for code, the first stop marker outside strings and comments."""
    if block == "final_answer":
        answer_end = text.find("</final_answer>", start)
        return len(text) if answer_end == -1 else answer_end
    for token in PYTHON_STRING_OR_STOP.finditer(text, start):
        if token.group("stop"):
            return token.start()
    return len(text)

def end_of_closed_section(text, section_opening, unclosed_sections):
    """Returns where the section opened by the section_opening match ends, or None, recording it in unclosed_sections
    so that later openings of an unclosed section cost no search."""
    section = section_opening.group("section").lower()
    if section in unclosed_sections:
        return None
    section_end = re.compile(f"</{section}>", re.IGNORECASE).search(text, section_opening.end())
    if section_end:
        return section_end.end()
    unclosed_sections.add(section)
    return None

def first_real_block_opening(text):
    """Returns the match of the first code block's opening tag outside files and reasoning sections, or None,
    and where the first bare stop marker before it is, or None."""
    position = 0
    stray_stop = None
    unclosed_sections = set()
    while tag := OUTSIDE_BLOCK_TAG.search(text, position):
        position = tag.end()
        if tag.group("stray_stop"):
            stray_stop = tag.start() if stray_stop is None else stray_stop
        elif tag.group("section"):
            position = end_of_closed_section(text, tag, unclosed_sections) or position
        else:
            return tag, stray_stop
    return None, stray_stop

def end_output_at_real_stop(text):
    """Returns text cut at its first real stop and with that block closed, ignoring markers in files, reasoning sections
    and Python strings. A bare marker cuts too, unless a code block opens after it."""
    opening, stray_stop = first_real_block_opening(text)
    if opening:
        block = opening.group("code_block")
        return text[:end_of_block(text, opening.end(), block)] + f"</{block}>"
    return text if stray_stop is None else text[:stray_stop]

def hide_tag_brackets(text):
    """Returns text with every "<" replaced by HIDDEN_TAG_BRACKET."""
    return text.replace("<", HIDDEN_TAG_BRACKET)

def restore_tag_brackets(text):
    """Returns text with every HIDDEN_TAG_BRACKET back to "<"."""
    return text.replace(HIDDEN_TAG_BRACKET, "<")

def hide_python_string_tags(code):
    """Returns code with the tags hidden in its Python strings. Closed sections are skipped so that an apostrophe in
    their text does not start a string that swallows their closing tag."""
    parts = []
    copied_up_to = 0
    position = 0
    unclosed_sections = set()
    while token := SECTION_OPENING_OR_PYTHON_STRING.search(code, position):
        position = token.end()
        if token.group("section"):
            position = end_of_closed_section(code, token, unclosed_sections) or position
        elif token.group().startswith(('"', "'")):
            parts += [code[copied_up_to:token.start()], hide_tag_brackets(token.group())]
            copied_up_to = token.end()
    return "".join(parts) + code[copied_up_to:]

def hide_tags_in_block_literals(text):
    """Returns the cut text with the tags hidden in its block's literal text: all of a final answer, or a code block's Python strings."""
    opening, _ = first_real_block_opening(text)
    if opening is None:
        return text
    block = opening.group("code_block")
    body_end = end_of_block(text, opening.end(), block)
    body = text[opening.end():body_end]
    hidden_body = hide_tag_brackets(body) if block == "final_answer" else hide_python_string_tags(body)
    return text[:opening.end()] + hidden_body + text[body_end:]

def escape_for_triple_quoted_string(text):
    """Returns text that reads back unchanged inside a triple-double-quoted Python string, keeping its real newlines.
    The word final_answer is hex-escaped because fix_final_answer_code renames it even inside strings."""
    return text.translate(TRIPLE_QUOTED_STRING_ESCAPES).replace("final_answer", r"\x66inal_answer")

def final_answer_string_body(answer_text):
    """Returns answer_text for pasting between triple quotes, with its include tags renamed to the _as_text form."""
    body_parts = []
    copied_up_to = 0
    for include_tag in FINAL_ANSWER_INCLUDE_TAG.finditer(answer_text):
        tag_name, tag_content = include_tag.groups()
        body_parts.append(escape_for_triple_quoted_string(answer_text[copied_up_to:include_tag.start()]))
        body_parts.append(f"<{tag_name}_as_text>{tag_content}</{tag_name}_as_text>")
        copied_up_to = include_tag.end()
    body_parts.append(escape_for_triple_quoted_string(answer_text[copied_up_to:]))
    return "".join(body_parts)

def strip_trailing_tags(code):
    """
    Removes any XML/HTML-like tags (e.g. </parameter>, </invoke>, <br>) that
    appear at the very end of a code block. Some models add stray tags just
    before closing </runcode>; this cleans the extracted code before execution.
    Only the tail of the text is touched, so tags inside strings stay intact.
    """
    if not code or not isinstance(code, str):
        return code
    pattern = re.compile(r'</?[A-Za-z_][\w\-]*\s*/?>\s*$')
    result = code.rstrip()
    while True:
        match = pattern.search(result)
        if not match:
            break
        result = result[:match.start()].rstrip()
    return result

def bp_parse_code_blobs(text: str) -> str:
    """Extract code blocs from the LLM's output.

    If a valid code block is passed, it returns it directly.

    Args:
        text (`str`): LLM's output text to parse.

    Returns:
        `str`: Extracted code block.

    Raises:
        ValueError: If no valid code block is found in the text.
    """
    pattern = CODE_BLOCK_OPEN + "(.*?)" + CODE_BLOCK_CLOSE + r"|```(?:python|py)\b\s*(.*?)```<end_code>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        blocks = (tagged or fenced for tagged, fenced in matches)
        return "\n\n".join(MARKDOWN_FENCE_AROUND_CODE.sub(r"\1", block.strip()) for block in blocks)
    # Maybe the LLM outputted a code blob directly
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass

    raise ValueError(
        dedent(
            """
            Your code snippet is invalid
            """
        ).strip()
    )
