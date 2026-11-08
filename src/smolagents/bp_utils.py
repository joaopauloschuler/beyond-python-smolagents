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
MARKDOWN_FENCE_AROUND_CODE = re.compile(r"\A```(?:python|py)\b\s*(.*?)\s*```\Z", re.DOTALL)

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
