import time
import ast
import re
from textwrap import dedent
import os
import glob
import shutil

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
    pattern = r"```(?:py|python)\s*(.*?)```<end_code>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    # Maybe the LLM outputted a code blob directly
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass

    raise ValueError(
        dedent(
            f"""
            Your code snippet is invalid
            """
        ).strip()
    )