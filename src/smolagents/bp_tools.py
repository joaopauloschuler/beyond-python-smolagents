# beyond python tools

from .tools import tool, Tool
from .default_tools import VisitWebpageTool
from .models import ChatMessage, MessageRole
import difflib
import os
import shutil
import subprocess
import shlex
import re
import textwrap
from slugify import slugify

_HAS_PRLIMIT = shutil.which("prlimit") is not None

RESTART_CHAT_TXT = """Use this sub assistant as much as you can with the goal to save your own context.
You can restart the chat by setting restart_chat to True.
For independent or new tasks to this sub assistant, you should use “restart_chat = True” so the context of this sub assistant will be smaller.
If you need to ask for more details, when asking for more details, you should set restart_chat to False.
Setting restart_chat to False is particularly useful when this sub assistant replies to you “I have completed the task” without providing any further information. In this case, you can ask for the missing details with “restart_chat to False”.
"""

DEFAULT_SOURCE_CODE_EXTENSIONS:tuple = (
        '.ada',
        '.asm', '.s',
        '.c', '.cc', '.cs', '.cpp', '.h', '.hpp', '.go', '.rs', '.swift',
        '.cob', '.cbl',
        '.dart', '.lua',
        '.f', '.f90',
        '.hs', '.ml', '.mli', '.fs', '.fsx', '.clj', '.cljs', '.scm', '.lisp',
        '.html', '.htm', '.js', '.css', '.ts', '.tsx', '.jsx',
        '.java', '.kt', '.kts', '.scala',
        '.pas', '.inc',  '.pp', '.lpr', '.dpr', '.lfm', '.dfm', 
        '.php', 
        '.py', '.ipynb',
        '.rb','.pl', '.pm','.sh', '.bash', '.ps1','.bat', '.cmd',        
        '.r', '.R', '.m', '.sql',
        '.txt', '.csv', '.md',
        '.toml', '.ini', '.cfg',
        '.xml', '.json', '.yml', '.yaml',
    )

@tool
def save_string_to_file(content: str, filename: str) -> bool:
    """
    Saves the given content to the specified file.
    Args:
      content: str
      filename: str
    """
    with open(filename, "w") as text_file:
      text_file.write(content)
    return True

@tool
def append_string_to_file(content: str, filename: str) -> bool:
    """
    Appends the given content to the specified file.
    Args:
      content: str
      filename: str
    """
    with open(filename, "a") as text_file:
      text_file.write(content)
    return True

@tool
def load_string_from_file(filename: str) -> str:
    """
Loads the content from the specified file.
For saving and printing a file, just enclose your text into the <savetofile></savetofile>:
<example>
<savetofile filename="another_file.csv">
header1,header2
value1,value2
value3,value4
</savetofile>
<runcode>
# print the content of another_file.csv
print(load_string_from_file(filename="another_file.csv"))
</runcode>
</example>

    Args:
      filename: str
    """
    content = ''
    if os.path.isfile(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # Try a different common encoding if utf-8 fails
                with open(filename, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                # If both common encodings fail, report the error
                raise ValueError(f"Could not read file {filename} due to encoding issues or other errors: {e}")
    return content

@tool
def copy_file(source_filename: str, dest_filename: str) -> bool:
    """
    Copy the source_filename into the dest_filename.
    Args:
      source_filename: str
      dest_filename: str
    """
    save_string_to_file(load_string_from_file(source_filename), dest_filename)
    return True

@tool
def replace_on_file(filename: str, old_value: str, new_value: str) -> str:
    """
    Replace the old_value with the new_value in the filename.
    This function is useful for fixing source code directly on file.
    It returns the updated file.
    This is its implementation:
    str_code = load_string_from_file(filename)
    new_code = str_code.replace(old_value, new_value)
    save_string_to_file(new_code, filename)
    return new_code
    Args:
      filename: str
      old_value: str
      new_value: str
    """
    str_code = load_string_from_file(filename)
    new_code = str_code.replace(old_value, new_value)
    save_string_to_file(new_code, filename)
    return new_code

@tool
def replace_on_file_with_files(filename: str, file_with_old_value: str, file_with_new_value: str) -> str:
    """
    Replace the content from the file_with_old_value with the content from the file_with_new_value in the filename.
    This function is useful for fixing source code directly on the file specified by filename.
    It returns the updated file.

    This is its implementation:
    str_code = load_string_from_file(filename)
    old_value = load_string_from_file(file_with_old_value)
    new_value = load_string_from_file(file_with_new_value)
    new_code = str_code.replace(old_value, new_value)
    save_string_to_file(new_code, filename)
    return new_code

In the case that you need to replace strings in an existing file, you can do it using the replace_on_file tool. This is an example:
<example>
<savetofile filename="tmp1.txt">
hello world
</savetofile>
<savetofile filename="tmp2.txt">
hello home!
</savetofile>
<savetofile filename="test.txt">
Hey! hello world
</savetofile>
<runcode>
replace_on_file_with_files('test.txt', 'tmp1.txt', 'tmp2.txt')
</runcode>
</example>

The expected file content of test.txt after the above is "Hey! hello home!".

    Args:
      filename: str
      file_with_old_value: str
      file_with_new_value: str
    """
    str_code = load_string_from_file(filename)
    old_value = load_string_from_file(file_with_old_value)
    new_value = load_string_from_file(file_with_new_value)
    new_code = str_code.replace(old_value, new_value)
    save_string_to_file(new_code, filename)
    return new_code

@tool
def get_file_size(filename: str) -> int:
    """
    Returns the size of the file in bytes.
    Args:
      filename: str
    """
    if os.path.isfile(filename):
      return os.path.getsize(filename)
    return 0

@tool
def is_file(filename: str) -> bool:
    """
    Returns true if filename is a file.
    Implemented as "os.path.isfile(filename)".
    Args:
      filename: str
    """
    return os.path.isfile(filename)

def remove_after_markers(text, stop_sequences=["</runcode>", "</code>", "Calling tools:"], after_first=True):
    """
    Args:
        text (str): The input string to process
        stop_sequences (list): List of strings to search for as stop markers
        after_first (bool): True means that it removes after the first. False means that it removes after last.

    Returns:
        str: The string with content after the last marker removed
    """
    if not text or not isinstance(text, str):
        return text

    if not stop_sequences:
        return text

    # Find positions of LAST occurrence of all stop sequences
    valid_positions = []

    for sequence in stop_sequences:
        if (after_first):
            pos = text.find(sequence)
        else:
            pos = text.rfind(sequence)
        if pos != -1:
            valid_positions.append(pos)

    if valid_positions:
        cut_position = min(valid_positions)  # Use min for the first occurrence
        return text[:cut_position]

    return text

@tool
def force_directories(file_path: str) -> None:
    """
    Extracts the directory path from a full file path and creates
    the directory structure if it does not already exist.

    Args:
        file_path: The full path to a file (e.g., '/path/to/some/directory/file.txt').
                   Can be a relative or absolute path.
    """
    # Use os.path.dirname() to get the directory part of the path.
    # This works for both files and directories (if path ends with a slash).
    directory_path = os.path.dirname(file_path)

    # If the path ends with a directory separator or refers to the current
    # directory or root, dirname might return an empty string or '.'.
    # os.makedirs handles these cases correctly.
    # os.makedirs() with exist_ok=True will create the directories recursively
    # if they don't exist, and will do nothing if they already exist.
    if directory_path: # Only attempt to create if directory_path is not empty
      os.makedirs(directory_path, exist_ok=True)

@tool
def run_os_command(str_command: str, timeout: int = 60, max_memory:int = 274877906944) -> str:
    """
Runs an OS command and returns the output.
This implementation uses Popen with shell=False.

For finding files in the file system, use this example:
<example>
<runcode>
print(run_os_command('find / -type f -iname "UTF8P*" 2>/dev/null'))
</runcode>
</example>

You can use run_os_command to run php code. This is an example:
<example>
<savetofile filename="hello.php">
<?php echo "hello"; ?>
</savetofile>
<runcode>
# 60 seconds of timeout and 512MB of max memory
print(run_os_command("php hello.php", timeout=60, max_memory=512*1024*1024))
</runcode>
</example>

As you can see in the above command, you can use any computer language that is available in the system. If it is not, you can install it using the run_os_command tool.

    Args:
      str_command: str
      timeout: int seconds
      max_memory: int bytes
    """
    if max_memory > 0 and _HAS_PRLIMIT:
        command = shlex.split("prlimit --as="+str(max_memory)+" "+str_command)
    else:
        command = shlex.split(str_command)
    result = ""
    outs = None
    errs = None
    try:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        try:
            outs, errs = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            result += "ERROR: timeout has expired. "
            proc.kill()
            outs, errs = proc.communicate()
        except Exception:
            proc.kill()
            outs, errs = proc.communicate()
    except Exception as e:
        result += f"ERROR: {e}"

    if outs is not None: result += outs.decode('utf-8')
    if errs is not None: result += errs.decode('utf-8')
    return result

@tool
def print_source_code_lines(filename: str, start_line: int, end_line: int) -> None:
  """ 
  This tool prints the lines from the start_line to the end_line of the file filename.
  In combination with get_line_from_file, this tool is useful for finding bugs in the source code.

  Args:
    filename: str The path to the text file.
    start_line: int
    end_line: int
  """
  file_content = load_string_from_file(filename)
  lines = file_content.splitlines()
  print("Content of " + filename + " from line "+str(start_line)+" to line " +str(end_line) )
  for i in range(start_line-1, end_line):
      print(f"{i+1}: {lines[i]}")

@tool
def get_file_lines(filename: str) -> int:
  """ 
  get_file_lines returns the number of lines in a text file.

  Args:
    filename: str The path to the text file.
  """
  file_content = load_string_from_file(filename)
  lines = file_content.splitlines()
  return len(lines)

@tool
def get_line_from_file(file_name: str, line_number: int) -> str:
    """
    Reads a specified line from a text file.
    This fuction was coded by Gemini 2.5Flash.

    This function is good for finding the line where the compiler gave an error. As per example:
Free Pascal Compiler version 3.2.2+dfsg-9ubuntu1 [2022/04/11] for x86_64
Copyright (c) 1993-2021 by Florian Klaempfl and others
Target OS: Linux for x86-64
Compiling solution1/src/jpmmath.pas
jpmmath.pas(301,19) Warning: function result variable of a managed type does not seem to be initialized

In the above example, to find the error, you can call:
<runcode>
print(get_line_from_file('solution1/src/jpmmath.pas', 301))
</runcode>

    Args:
        file_name: str The path to the text file.
        line_number: int The 1-based index of the line to retrieve.
                           For example, 1 for the first line, 2 for the second, etc.

    Returns:
        str: The content of the specified line, with leading/trailing whitespace
             (including the newline character) removed.

    Raises:
        ValueError: If file_name is not a valid string, or line_number is not
                    a positive integer.
        FileNotFoundError: If the specified file does not exist.
        IndexError: If the line_number is out of the bounds of the file
                    (e.g., requesting line 10 from a 5-line file).
        IOError: For other potential I/O errors during file reading.
    """
    # 1. Input Validation
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("file_name must be a non-empty string.")
    if not isinstance(line_number, int):
        raise TypeError("line_number must be an integer.")
    if line_number <= 0:
        raise ValueError("line_number must be a positive integer (1-based).")

    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            # Enumerate starts from 0 by default, so we compare with line_number - 1
            for current_line_index, line_content in enumerate(f):
                if current_line_index == line_number - 1:
                    # rstrip() removes trailing whitespace, including the newline char
                    return line_content.rstrip('\n\r') # Handles both \n and \r\n

            # If the loop finishes, it means the requested line_number was
            # beyond the total number of lines in the file.
            raise IndexError(f"Line number {line_number} is out of bounds for file '{file_name}'.")

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{file_name}' not found.")
    except IOError as e:
        # Catch other potential I/O errors (e.g., permission denied)
        raise IOError(f"An I/O error occurred while reading file '{file_name}': {e}")
    except Exception as e:
        # Catch any other unexpected errors
        raise Exception(f"An unexpected error occurred: {e}")

@tool
def replace_line_in_file(file_name: str, line_number: int, new_content: str) -> None:
    """
    Replaces a specified line in a text file with new content.

    This function reads the entire file, replaces the specified line with new content,
    and writes the modified content back to the file. If new_content contains newline
    characters, it will be split into multiple lines that replace the single original line.

    Example usage:
        # Replace line 301 with a corrected version
        replace_line_in_file('solution1/src/jpmmath.pas', 301, 'function Result: Integer;')
        
        # Replace line 10 with multiple lines
        replace_line_in_file('test.txt', 10, 'Line 10a\\nLine 10b\\nLine 10c')
        
        # Fix a compiler error
        error_line = get_line_from_file('solution1/src/jpmmath.pas', 301)
        print(f"Original: {error_line}")
        replace_line_in_file('solution1/src/jpmmath.pas', 301, '  Result := 0;')

    Args:
        file_name: str The path to the text file to modify.
        line_number: int The 1-based index of the line to replace.
                         For example, 1 for the first line, 2 for the second, etc.
        new_content: str The new content to replace the line with.
                        Can contain newline characters (\\n) to insert multiple lines.

    Returns:
        None: The function modifies the file in place.

    Raises:
        ValueError: If file_name is not a valid string or line_number is not
                   a positive integer.
        TypeError: If line_number is not an integer or new_content is not a string.
        FileNotFoundError: If the specified file does not exist.
        IndexError: If the line_number is out of the bounds of the file.
        IOError: For other potential I/O errors during file reading or writing.
        PermissionError: If the program doesn't have write permission for the file.
    """
    # 1. Input Validation
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("file_name must be a non-empty string.")
    if not isinstance(line_number, int):
        raise TypeError("line_number must be an integer.")
    if line_number <= 0:
        raise ValueError("line_number must be a positive integer (1-based).")
    if not isinstance(new_content, str):
        raise TypeError("new_content must be a string.")

    try:
        # 2. Read the entire file
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 3. Validate line_number is within bounds
        if line_number > len(lines):
            raise IndexError(
                f"Line number {line_number} is out of bounds for file '{file_name}' "
                f"(file has {len(lines)} lines)."
            )

        # 4. Prepare the replacement - ensure it ends with \n for proper file format
        if not new_content.endswith('\n'):
            new_content += '\n'

        # 5. Replace the specified line (convert to 0-based index)
        # If new_content contains \n characters, it will be written as-is,
        # effectively replacing one line with potentially multiple lines
        lines[line_number - 1] = new_content

        # 6. Write the modified content back to the file
        with open(file_name, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{file_name}' not found.")
    except PermissionError as e:
        raise PermissionError(f"Permission denied: Cannot write to file '{file_name}': {e}")
    except IOError as e:
        raise IOError(f"An I/O error occurred while accessing file '{file_name}': {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

@tool
def insert_lines_into_file(file_name: str, line_number: int, new_content: str) -> None:
    """
    Inserts new content before a specified line in a text file.

    This function reads the entire file, inserts new content before the specified line,
    and writes the modified content back to the file. The original line at line_number
    and all subsequent lines are shifted down. If new_content contains newline
    characters, multiple lines will be inserted.

    Example usage:
        # Insert a new line before line 301
        insert_lines_into_file('solution1/src/jpmmath.pas', 301, 'var Result: Integer;')
        
        # Insert multiple lines before line 10
        insert_lines_into_file('test.txt', 10, 'Line 9a\\nLine 9b\\nLine 9c')
        
        # Insert at the beginning of the file
        insert_lines_into_file('test.txt', 1, '// File header comment')
        
        # Append to the end of the file (if file has 100 lines, use 101)
        insert_lines_into_file('test.txt', 101, 'New last line')

    Args:
        file_name: str The path to the text file to modify.
        line_number: int The 1-based index before which to insert.
                         For example, 1 inserts at the beginning,
                         2 inserts before the current line 2 (making it the new line 2),
                         len(lines)+1 appends to the end of the file.
        new_content: str The content to insert.
                        Can contain newline characters (\\n) to insert multiple lines.

    Returns:
        None: The function modifies the file in place.

    Raises:
        ValueError: If file_name is not a valid string or line_number is not
                   a positive integer.
        TypeError: If line_number is not an integer or new_content is not a string.
        FileNotFoundError: If the specified file does not exist.
        IndexError: If the line_number is out of the valid range (1 to len(lines)+1).
        IOError: For other potential I/O errors during file reading or writing.
        PermissionError: If the program doesn't have write permission for the file.
    """
    # 1. Input Validation
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("file_name must be a non-empty string.")
    if not isinstance(line_number, int):
        raise TypeError("line_number must be an integer.")
    if line_number <= 0:
        raise ValueError("line_number must be a positive integer (1-based).")
    if not isinstance(new_content, str):
        raise TypeError("new_content must be a string.")

    try:
        # 2. Read the entire file
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 3. Validate line_number is within bounds
        # For insertion, valid range is 1 to len(lines)+1
        # line_number = len(lines)+1 means append to end
        if line_number > len(lines) + 1:
            raise IndexError(
                f"Line number {line_number} is out of bounds for insertion in file '{file_name}' "
                f"(file has {len(lines)} lines, valid range is 1-{len(lines) + 1})."
            )

        # 4. Prepare the content to insert - ensure it ends with \n
        if not new_content.endswith('\n'):
            new_content += '\n'

        # 5. Insert the new content before the specified line (convert to 0-based index)
        # If new_content contains \n characters, it will be written as-is,
        # effectively inserting multiple lines
        lines.insert(line_number - 1, new_content)

        # 6. Write the modified content back to the file
        with open(file_name, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{file_name}' not found.")
    except PermissionError as e:
        raise PermissionError(f"Permission denied: Cannot write to file '{file_name}': {e}")
    except IOError as e:
        raise IOError(f"An I/O error occurred while accessing file '{file_name}': {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

@tool
def run_php_file(filename: str, timeout: int = 60) -> str:
    """
Runs a PHP file and returns the output.
To run PHP code, follow tis an example:
<example>
<savetofile filename="hello.php">
<?php echo "hello"; ?>
</savetofile>
<runcode>
print(run_php_file("hello.php", timeout=60))
</runcode>
</example>

Args:
    filename: str
    timeout: int
"""
    return run_os_command("php " + filename, timeout)

@tool
def compile_and_run_pascal_code(pasfilename: str, timeout: int = 60) -> str:
  """
Compiles and runs pascal code. pasfilename contains the filename to be compiled.
If you need to pass additional parameters such as to include existing units, you can include into the pasfilename parameter.
This is an example to compile a file named myfile.pas with the units from neural-api/neural:
compile_and_run_pascal_code('-Funeural-api/neural/ myfile.pas', 120)

This is an example for running pascal code:
<example>
<savetofile filename="hello.pas">
program hello;
begin
WriteLn('Hello');
end.
</savetofile>
<runcode>
compile_and_run_pascal_code("hello.pas", timeout=60)
</runcode>
</example>

    Args:
      pasfilename: str
      timeout: int
  """
  filename = 'compiled'
  if os.path.exists(filename):
      os.remove(filename)
  print(run_os_command("fpc -O3 -Mobjfpc "+pasfilename+' -o'+filename, timeout=timeout))
  if os.path.exists(filename):
    print(run_os_command("./compiled", timeout=timeout))
  else:
    print('Compilation error.')

@tool
def remove_pascal_comments_from_string(code_string: str) -> str:
    """
    Remove all comments from a Delphi/Pascal code string.
    
    Handles:
    - Single-line comments (//)
    - Brace comments ({ }) with nesting
    - Parenthesis-asterisk comments ((* *))
    - Preserves comment-like text inside string literals

    Args:
      code_string: str
    """
    result = []
    i = 0
    length = len(code_string)

    # State tracking
    in_string = False
    string_char = None  # ' or "
    brace_comment_depth = 0
    in_paren_comment = False
    in_line_comment = False

    while i < length:
        char = code_string[i]
        next_char = code_string[i + 1] if i + 1 < length else None

        # Handle string literals first
        if not (brace_comment_depth > 0 or in_paren_comment or in_line_comment):
            if not in_string and char in ["'", '"']:
                in_string = True
                string_char = char
                result.append(char)
                i += 1
                continue
            elif in_string and char == string_char:
                # Check for escaped quote (double quote in Pascal)
                if next_char == string_char:
                    result.append(char)
                    result.append(next_char)
                    i += 2
                    continue
                else:
                    in_string = False
                    string_char = None
                    result.append(char)
                    i += 1
                    continue

        # If we're in a string, just copy characters
        if in_string:
            result.append(char)
            i += 1
            continue

        # Handle end of line comment
        if in_line_comment:
            if char in ['\n', '\r']:
                in_line_comment = False
                result.append(char)  # Preserve newlines
            i += 1
            continue

        # Handle end of (* *) comment
        if in_paren_comment:
            if char == '*' and next_char == ')':
                in_paren_comment = False
                i += 2
                continue
            i += 1
            continue

        # Handle end of { } comment
        if brace_comment_depth > 0:
            if char == '}':
                brace_comment_depth -= 1
            elif char == '{':
                brace_comment_depth += 1  # Handle nesting
            i += 1
            continue

        # Check for start of comments (only when not in any comment or string)
        if char == '/' and next_char == '/':
            in_line_comment = True
            i += 2
            continue
        elif char == '{':
            brace_comment_depth = 1
            i += 1
            continue
        elif char == '(' and next_char == '*':
            in_paren_comment = True
            i += 2
            continue

        # Normal character - add to result
        result.append(char)
        i += 1

    return ''.join(result)

@tool
def source_code_to_string(folder_name: str, 
    allowed_extensions: tuple = DEFAULT_SOURCE_CODE_EXTENSIONS,
    remove_pascal_comments: bool = False,
    exclude_list: tuple = ('excluded_folder','excluded_file.pas')) -> str:
    """
    Scans a folder and subfolders for specific source code file types (.py, .txt, .pas, .inc, .md),
    concatenates their content into a single string with XML-like tags,
    and orders the files in the output string (.md first, then .txt, then others),
    using the base filename in the tag.

    Includes robust error handling for file reading and deterministic sorting.

    Args:
        folder_name: The path to the root folder to scan.
        allowed_extensions: tuple of allowed file extensions to scan.
        remove_pascal_comments: if true, removes pascal comments
        exclude_list: list of files or folders that will not be included
    Returns:
        A single string containing the concatenated content of the scanned files,
        formatted with <file filename="...">...</file> tags, or an empty string
        if the folder does not exist or no relevant files are found.
    """
    if not os.path.isdir(folder_name):
        print(f"Error: Folder '{folder_name}' not found.")
        return ""

    relevant_files_info = []

    for root, _, files in os.walk(folder_name):
        for filename in files:
            filepath = os.path.join(root, filename)
            _, file_extension = os.path.splitext(filename)
            file_extension_lower = file_extension.lower()
            subfolders_tuple = tuple(root.split('/'))
            not_exclude = True
            for item in subfolders_tuple:
                if item in exclude_list:
                    not_exclude = False
                    break  # Exit the loop once a match is found

            if not_exclude and \
                (file_extension_lower in allowed_extensions) and \
                (not (filename in exclude_list)):
                # Store full path for sorting, base filename for output tag, and extension
                relevant_files_info.append({
                    'filepath': filepath,
                    'filename': filename, # Use base filename for the tag
                    'extension': file_extension_lower
                })

    # Custom sorting key: .md files first (0), then .txt files (1), then others (2).
    # Within each group, sort alphabetically by full path for consistency (deterministic).
    def sort_key(file_info):
        extension = file_info['extension']
        filepath = file_info['filepath']
        if extension == '.md':
            primary_key = 0
        elif extension == '.txt':
            primary_key = 1
        else:
            primary_key = 2
        return (primary_key, filepath)

    # Sort the list of file info dictionaries
    relevant_files_info.sort(key=sort_key)

    output_string_parts = []

    for file_info in relevant_files_info:
        filepath = file_info['filepath']
        filename_for_tag = filepath.replace(folder_name+'/','')
        content = ""
        try:
            # Attempt to read file content, trying multiple common encodings
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                 try:
                    # Try a different common encoding if utf-8 fails
                    with open(filepath, 'r', encoding='latin-1') as f:
                        content = f.read()
                 except Exception as e:
                    # If both common encodings fail, report the error
                    print(f"Could not read file {filepath} due to encoding issues or other errors.")
                    content = f"Error reading file content (encoding or other issue): {e}"

        except FileNotFoundError:
             # This should ideally not happen since os.walk found it, but included for robustness
             print(f"Error: File not found unexpectedly: {filepath}")
             content = "Error: File not found unexpectedly."

        except Exception as e:
            # Catch any other potential reading errors
            print(f"An unexpected error occurred while reading file {filepath}: {e}")
            content = f"An unexpected error occurred while reading: {e}"

        if (remove_pascal_comments):
            content = remove_pascal_comments_from_string(content)
        # Format the content block using the base filename
        formatted_block = f'<file filename="{filename_for_tag}">\n{content}\n</file>'
        output_string_parts.append(formatted_block)


    # Join all formatted blocks with a newline separator between blocks
    return '\n'.join(output_string_parts)


@tool
def string_to_source_code(string_with_files: str, output_base_dir: str = '.', overwrite: bool = True, verbose: bool = True) -> None:
    """
    Parses a string containing concatenated file content with <file filename="...">...</file> tags,
    and recreates the files and directories in a specified base directory.
    This function does the opposite work of the function source_code_to_string.

    Args:
        string_with_files: The input string containing file data.
        output_base_dir: The base directory where files should be saved. Defaults to the current directory.
        overwrite: If True, overwrite existing files. If False, skip saving the file if it already exists.
        verbose: If True, print status and error messages during processing.
    """
    if verbose:
        print("Starting file reconstruction process...")
        print(f"Target output base directory: {os.path.abspath(output_base_dir)}")
        print(f"Overwrite existing files: {overwrite}")

    # Regex to find file blocks.
    # - <file filename="([^"]+?)"> : Matches the opening tag and captures the filename attribute
    #   - ([^"]+?) : Group 1, captures one or more characters that are NOT a double quote (non-greedy)
    # - (.*?) : Group 2, captures the content (non-greedy). Uses re.DOTALL to match newlines.
    # - </file> : Matches the closing tag
    # This regex is robust for filenames within double quotes and captures the content exactly
    # as it appears between the tags (including any surrounding newlines added by the source function).
    file_pattern = re.compile(r'<file filename="([^"]+?)">\n(.*?)\n</file>', re.DOTALL)

    matches = file_pattern.findall(string_with_files)

    if not matches:
        if verbose:
            print("No file blocks found in the input string.")
        return

    if verbose:
        print(f"Found {len(matches)} file blocks.")

    successful_saves = 0
    skipped_saves = 0
    failed_saves = 0

    for relative_filepath, content in matches:
        # Construct the full output path using the specified base directory
        output_filepath = os.path.join(output_base_dir, relative_filepath)

        if verbose:
            print(f"\nProcessing file block for: {relative_filepath}")
            print(f"Target output path: {output_filepath}")

        # Ensure the directory exists for the output file
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir): # Only create directory if it's not the root and doesn't exist
            if verbose:
                print(f"Ensuring directory exists: {output_dir}")
            try:
                # exist_ok=True prevents an error if the directory already exists
                os.makedirs(output_dir, exist_ok=True)
                if verbose:
                    print(f"Directory ensured: {output_dir}")
            except OSError as e:
                if verbose:
                    print(f"Error creating directory {output_dir}: {e}")
                    print(f"Skipping file {output_filepath} due to directory creation error.")
                failed_saves += 1
                continue # Skip saving this file

        # Check if file exists and if overwrite is disabled
        if os.path.exists(output_filepath) and not overwrite:
            if verbose:
                print(f"File already exists and overwrite is False. Skipping file: {output_filepath}")
            skipped_saves += 1
            continue # Skip saving this file

        # Attempt to save file
        if verbose:
            print(f"Attempting to save file: {output_filepath}")
        try:
            # Use utf-8 encoding for writing
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            if verbose:
                print(f"Successfully saved file: {output_filepath}")
            successful_saves += 1
        except IOError as e:
            if verbose:
                print(f"Error writing file {output_filepath}: {e}")
            failed_saves += 1
        except Exception as e:
            if verbose:
                 print(f"An unexpected error occurred while writing file {output_filepath}: {e}")
            failed_saves += 1

    if verbose:
        print("\nFile reconstruction process finished.")
        print(f"Summary: {successful_saves} files saved successfully, {skipped_saves} skipped, {failed_saves} failed.")

@tool
def get_pascal_interface_from_file(filename: str, remove_pascal_comments: bool = False) -> str:
        """
        Returns the pascal interface of a pascal source code file.
        Args:
            filename: The pascal source code file name.
            remove_pascal_comments: if true, removes pascal comments.
        Returns:
            The pascal interface
        """
        return get_pascal_interface_from_code(load_string_from_file(filename), remove_pascal_comments)

@tool
def get_pascal_interface_from_code(content: str, remove_pascal_comments: bool = False) -> str:
        """
        Returns the interface of a pascal source code
        Args:
            content: The pascal source code.
            remove_pascal_comments: if true, removes pascal comments.
        Returns:
            The pascal interface
        """
        # --- Robust Extraction Logic (Stateful Character Parser from Solution 2) ---
        interface_section_content_chars = [] # Use a list to build the string efficiently
        is_interface_found = False
        is_implementation_found = False # Stop when implementation is found after interface

        in_curly_comment = False
        in_star_comment = False
        in_string = False
        in_line_comment = False

        i = 0
        while i < len(content):
            char = content[i]
            chars_left = len(content) - i

            # Check for end of line comment
            if in_line_comment and char == '\n':
                in_line_comment = False
                # Newline terminates the line comment state. It should be included if capturing.
                if is_interface_found and not is_implementation_found:
                     interface_section_content_chars.append(char)
                i += 1
                continue
            elif in_line_comment:
                 # Inside line comment, do not process keywords or toggle states.
                 # Include character in output if capturing interface section.
                 if is_interface_found and not is_implementation_found:
                      interface_section_content_chars.append(char)
                 i += 1
                 continue

            # State transitions for block comments and strings (only if not in line comment)
            if not in_line_comment:
                # Handle string state (simplified: toggle on single quote)
                if char == "'":
                    in_string = not in_string
                    # Include the quote if capturing
                    if is_interface_found and not is_implementation_found:
                         interface_section_content_chars.append(char)
                    i += 1
                    continue

                # Handle } for curly comment end
                if in_curly_comment and char == '}':
                    in_curly_comment = False
                    # Include the bracket if capturing
                    if is_interface_found and not is_implementation_found:
                         interface_section_content_chars.append(char)
                    i += 1
                    continue

                # Handle *} for star comment end
                if in_star_comment and chars_left >= 2 and content[i:i+2] == '*}':
                     in_star_comment = False
                     # Include the bracket if capturing
                     if is_interface_found and not is_implementation_found:
                          interface_section_content_chars.extend(content[i:i+2])
                     i += 2
                     continue

                # Handle { for curly comment start
                if char == '{' and not in_star_comment and not in_string: # { cannot start a curly comment inside star comment or string
                    # Check if it's a star comment {*
                    if chars_left >= 2 and content[i+1] == '*':
                         # This is {*, handled below
                         pass # Let the next check handle {*
                    else: # This is {
                        in_curly_comment = True
                        # Include the bracket if capturing
                        if is_interface_found and not is_implementation_found:
                             interface_section_content_chars.append(char)
                        i += 1
                        continue

                # Handle {* for star comment start
                if chars_left >= 2 and content[i:i+2] == '{*' and not in_curly_comment and not in_string: # {* cannot start inside curly comment or string
                     in_star_comment = True
                     # Include the marker if capturing
                     if is_interface_found and not is_implementation_found:
                          interface_section_content_chars.extend(content[i:i+2])
                     i += 2
                     continue

                # Handle // for line comment start
                if chars_left >= 2 and content[i:i+2] == '//' and not in_curly_comment and not in_star_comment and not in_string:
                    in_line_comment = True
                    # Include the marker if capturing
                    if is_interface_found and not is_implementation_found:
                         interface_section_content_chars.extend(content[i:i+2])
                    i += 2
                    continue

            # --- Keyword Detection (only when not in comment or string or line comment) ---
            # Only check for keywords if none of the comment/string states are active
            if not in_curly_comment and not in_star_comment and not in_string and not in_line_comment:
                # Check for 'interface' keyword
                if not is_interface_found and chars_left >= len('interface') and content[i:i+len('interface')].lower() == 'interface':
                     # Check for word boundary - simplified check using isalnum() and _
                     is_word_boundary_before = (i == 0 or not content[i-1].isalnum() and content[i-1] != '_')
                     is_word_boundary_after = (i + len('interface') == len(content) or not content[i+len('interface')].isalnum() and content[i+len('interface')] != '_')

                     if is_word_boundary_before and is_word_boundary_after:
                         is_interface_found = True
                         # Start capturing content *after* the keyword 'interface'
                         # Skip the keyword itself
                         i += len('interface')
                         # Skip any immediate whitespace after the keyword
                         while i < len(content) and content[i].isspace():
                             i += 1
                         continue # Continue loop from the character after 'interface' + whitespace

                # Check for 'implementation' keyword (only if interface was found)
                if is_interface_found and not is_implementation_found and chars_left >= len('implementation') and content[i:i+len('implementation')].lower() == 'implementation':
                     # Check for word boundary
                     is_word_boundary_before = (i == 0 or not content[i-1].isalnum() and content[i-1] != '_')
                     is_word_boundary_after = (i + len('implementation') == len(content) or not content[i+len('implementation')].isalnum() and content[i+len('implementation')] != '_')

                     if is_word_boundary_before and is_word_boundary_after:
                        is_implementation_found = True
                        # Stop processing this file's content as we found the end marker
                        # Do NOT include 'implementation' keyword or anything after it
                        break

            # --- Content Capture ---
            # If we are inside the interface section AND the 'implementation' keyword hasn't been found yet,
            # append the current character to the output. This captures everything between 'interface' and 'implementation',
            # including comments and strings within that section.
            if is_interface_found and not is_implementation_found:
                 interface_section_content_chars.append(char)

            # Move to the next character if not handled by multi-char sequence above
            i += 1

        # --- End of Extraction Logic ---

        # Join the captured characters into a string
        interface_content = "".join(interface_section_content_chars).strip() # Strip leading/trailing whitespace
        if (remove_pascal_comments):
            interface_content = remove_pascal_comments_from_string(interface_content)
        return interface_content

@tool
def pascal_interface_to_string(folder_name: str, remove_pascal_comments: bool = False) -> str:
    """
    Scans a folder and subfolders for Pascal source code file types
    (.pas, .inc, .pp, .lpr, .dpr), extracts the content between the 'interface'
    and 'implementation' keywords (case-insensitive) using a robust stateful
    parser that correctly ignores content within comments and strings.
    Concatenates the extracted content into a single string with
    <pascal_interface filename="...">...</pascal_interface> tags.

    If an 'implementation' section is not found after 'interface', it extracts
    from 'interface' to the end of the file. If 'interface' is not found,
    it extracts nothing for that file's interface section. Handles basic
    encoding issues. Reports file reading errors within the output tags.

    Args:
        folder_name: The path to the root folder to scan.
        remove_pascal_comments: if true, removes pascal comments.
    Returns:
        A single string containing the concatenated interface sections,
        formatted with tags, or an empty string if the folder does not
        exist or no relevant files are found.
    """
    if not os.path.isdir(folder_name):
        # Print to execution log for debugging/info, but return empty string as per inspired function
        print(f"Error: Folder '{folder_name}' not found.")
        return ""

    relevant_files = []
    # Added .pp, .lpr, .dpr based on common Pascal file types
    allowed_extensions = ('.pas', '.inc', '.pp', '.lpr', '.dpr')

    for root, _, files in os.walk(folder_name):
        for filename in files:
            filepath = os.path.join(root, filename)
            _, file_extension = os.path.splitext(filename)
            file_extension_lower = file_extension.lower()

            if file_extension_lower in allowed_extensions:
                relevant_files.append(filepath)

    # Sort files alphabetically by full path for deterministic output
    relevant_files.sort()

    output_parts = []

    for filepath in relevant_files:
        # Determine the relative path for the tag filename attribute
        relative_filepath = os.path.relpath(filepath, folder_name)
        content = ""
        interface_content = ""
        read_error = None

        try:
            # Attempt to read file content, trying multiple common encodings
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                 try:
                    # Try a different common encoding if utf-8 fails
                    with open(filepath, 'r', encoding='latin-1') as f:
                        content = f.read()
                 except Exception as e:
                    read_error = f"Could not read file due to encoding issues or other errors: {e}"
                    print(f"Error reading file {filepath}: {read_error}") # Print to execution log


        except FileNotFoundError:
             # Should not happen based on os.walk, but included for robustness
             read_error = "File not found unexpectedly."
             print(f"Error reading file {filepath}: {read_error}") # Print to execution log

        except Exception as e:
            # Catch any other potential reading errors
            read_error = f"An unexpected error occurred while reading: {e}"
            print(f"An unexpected error occurred while reading file {filepath}: {e}") # Print to execution log

        if read_error:
            # If read failed, add an error tag and skip to the next file
             formatted_block = f'<pascal_interface filename="{relative_filepath}">\nError reading file: {read_error}\n</pascal_interface>'
             output_parts.append(formatted_block)
             continue # Skip to the next file

        interface_content = get_pascal_interface_from_code(content, remove_pascal_comments)
        # Construct the output block for this file using the tag format from Solution 1
        # Only add a block if 'interface' was actually found in the file
        if len(interface_content) > 10:
             formatted_block = f'<pascal_interface filename="{relative_filepath}">\n{interface_content}\n</pascal_interface>'
             output_parts.append(formatted_block)

    # Join all formatted blocks with a newline separator between blocks
    # Add an extra newline after each block for better readability in the final output
    return '\n\n'.join(output_parts)

@tool
def trim_right_lines(multi_line_string: str) -> str:
  """
  This function will do a right trim in all lines of the string.
  Args:
    multi_line_string: str
  """
  lines = multi_line_string.splitlines()
  # Trim only the right side of each line
  trimmed_lines = [line.rstrip() for line in lines]
  # Join the lines back together
  trimmed_string = '\n'.join(trimmed_lines)
  return trimmed_string

@tool
def trim_right_lines_in_file(filename: str) -> None:
  """
  This function will do a right trim in all lines of the file.
  Args:
    filename: str
  """
  multi_line_string = load_string_from_file(filename)
  save_string_to_file(trim_right_lines(multi_line_string), filename)

class Summarize(Tool):
    name = "summarize"
    description = """This subassistant will return the summary of a string.
"""+RESTART_CHAT_TXT
    inputs = {
        "text_str": {
            "type": "string",
            "description": "Input text to be summarized.",
        },
        "restart_chat": {
            "type": "boolean",
            "description": "When true, forgets the previous chat.",
            "nullable" : True
        }
    }
    output_type = "string"
    agent = None

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def forward(self, text_str: str, restart_chat: bool = True) -> str:
        task_str = 'Hello super-intelligence! Please provide a summary for the following as a string: '+ text_str
        result = self.agent.run(task_str, reset=restart_chat)
        return result

class SummarizeUrl(Tool):
    name = "summarize_url"
    description = """This subassistant will return the summary of a web page given its url as a string.
"""+RESTART_CHAT_TXT
    inputs = {
        "url": {
            "type": "string",
            "description": "url to be summarized.",
        },
        "restart_chat": {
            "type": "boolean",
            "description": "When true, forgets the previous chat.",
            "nullable" : True
        }
    }
    output_type = "string"
    agent = None

    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def forward(self, url: str, restart_chat: bool = True) -> str:
        LocalVistWebPageTool = VisitWebpageTool()
        task_str = 'Hello super-intelligence! Please write all the information in plain English text without tags from the following as a string (do not use python code except for the final answer): '+ LocalVistWebPageTool(url)[:15000]
        result = self.agent.run(task_str, reset=restart_chat)
        return result

class SummarizeLocalFile(Tool):
        name = "summarize_local_file"
        description = """This function will return the summary of a local file.
"""+RESTART_CHAT_TXT
        inputs = {
            "filename": {
                "type": "string",
                "description": "File in the file system.",
            },
            "restart_chat": {
                "type": "boolean",
                "description": "When true, forgets the previous chat.",
                "nullable" : True
            }
        }
        output_type = "string"
        agent = None

        def __init__(self, agent):
            super().__init__()
            self.agent = agent

        def forward(self, filename: str, restart_chat: bool = True) -> str:
            task_str = 'Hello super-intelligence! Please provide a summary for the following as a string (do not use python code except for the final answer): '+ load_string_from_file(filename)[:15000]
            result = self.agent.run(task_str, reset=restart_chat)
            return result

class Subassistant(Tool):
        name = "subassistant"
        description = """This assistant is similar to yourself in capability. It is called the sub assistant.
Check what the site https://cnn.com is about or
create a summary from the content of the file /content/README.md .
"""+RESTART_CHAT_TXT
        inputs = {
            "task_str": {
                "type": "string",
                "description": "Task description.",
            },
            "restart_chat": {
                "type": "boolean",
                "description": "When true, forgets the previous chat.",
                "nullable" : True
            }
        }
        output_type = "string"
        agent = None

        def __init__(self, agent):
            super().__init__()
            self.agent = agent

        def forward(self, task_str: str, restart_chat: bool = True) -> str:
            result = self.agent.run(task_str, reset=restart_chat)
            return result

class InternetSearchSubassistant(Tool):
        name = "internet_search_subassistant"
        description = """This assistant is similar to yourself in capability. It is called the internet search sub assistant.
This sub assistant is dedicated to internet searches.
"""+RESTART_CHAT_TXT
        inputs = {
            "task_str": {
                "type": "string",
                "description": "search topic.",
            },
            "restart_chat": {
                "type": "boolean",
                "description": "When true, forgets the previous chat.",
                "nullable" : True
            }
        }
        output_type = "string"
        agent = None

        def __init__(self, agent):
            super().__init__()
            self.agent = agent

        def forward(self, task_str: str, restart_chat: bool = True) -> str:
            local_task_str = """Hello super intelligence!
Please do an internet search regarding '"""+task_str+"""'.
Then, please reply with as much information as you can via final_answer('Hello, my findings are:...') .
In your answer, please include the references (links).
"""
            result = self.agent.run(local_task_str, reset=restart_chat)
            return result

class CoderSubassistant(Tool):
        name = "coder_subassistant"
        description = """This assistant is similar to yourself in capability. It is called the coder sub assistant.
"""+RESTART_CHAT_TXT
        inputs = {
            "task_str": {
                "type": "string",
                "description": "Coding task description.",
            },
            "restart_chat": {
                "type": "boolean",
                "description": "When true, forgets the previous chat.",
                "nullable" : True
            }
        }
        output_type = "string"
        agent = None

        def __init__(self, agent):
            super().__init__()
            self.agent = agent

        def forward(self, task_str: str, restart_chat: bool = True) -> str:
            local_task_str = """Hello super intelligence!
Please code '"""+task_str+"""'.
Then, please reply with your code via 
<final_answer>
# my code ...
...
</final_answer>
"""
            result = self.agent.run(local_task_str, reset=restart_chat)
            return result

class GetRelevantInfoFromFile(Tool):
        name = "get_relevant_info_from_file"
        description = """This sub assistant will return relevant information about relevant_about_str from a local file.
"""+RESTART_CHAT_TXT
        inputs = {
            "relevant_about_str": {
                "type": "string",
                "description": "What are we looking for.",
            },
            "filename": {
                "type": "string",
                "description": "File in the file system.",
            },
            "restart_chat": {
                "type": "boolean",
                "description": "When true, forgets the previous chat.",
                "nullable" : True
            }
        }
        output_type = "string"
        agent = None

        def __init__(self, agent):
            super().__init__()
            self.agent = agent

        def forward(self, relevant_about_str:str, filename: str, restart_chat: bool = True) -> str:
            task_str = """Hello super-intelligence! I have the following content in the tags <md></md>. This is the content:
<md>
"""+load_string_from_file(filename)[:128000*4]+"""
</md>
Please provide relevant information about """+relevant_about_str+""" from the above tags <md></md>.
Do not use python code except for the final answer - the output format must be a string.
You will provide the relevant information following this example:
<runcode>final_answer('This is what I have found: ... ')</runcode>.
"""
            result = self.agent.run(task_str, reset=restart_chat)
            return result

class GetRelevantInfoFromUrl(Tool):
        name = "get_relevant_info_from_url"
        description = """This sub assistant will return relevant information from an url.
"""+RESTART_CHAT_TXT
        inputs = {
            "relevant_about_str": {
                "type": "string",
                "description": "What are we looking for.",
            },
            "url": {
                "type": "string",
                "description": "URL from where information will be read.",
            },
            "restart_chat": {
                "type": "boolean",
                "description": "When true, forgets the previous chat.",
                "nullable" : True
            }
        }
        output_type = "string"
        agent = None

        def __init__(self, agent):
            super().__init__()
            self.agent = agent

        def forward(self, relevant_about_str:str, url: str, restart_chat: bool = True) -> str:
            LocalVistWebPageTool = VisitWebpageTool(128000*4)
            task_str = """Hello super-intelligence! I have the following md content in the tags <md></md>. This is the content:
<md>
"""+LocalVistWebPageTool(url)+"""
</md>
Please provide relevant information about """+relevant_about_str+""" from the above tags <md></md>.
Do not use python code except for the final answer - the output format must be a string.
You will provide the relevant information following this example:
<runcode>final_answer('This is what I have found: ... ')</runcode>.
"""
            result = self.agent.run(task_str, reset=restart_chat)
            return result

@tool
def get_files_in_folder(folder:str='solutions', fileext:str='.md') -> list:
  """
  This function will return a list of files in a folder with a given file extension.
  Args:
    folder: str
    fileext: str
  Returns:
    list: A list of filenames that match the specified extension
  """
  return [f for f in os.listdir(folder) if f.endswith(fileext)]

@tool
def create_filename(topic:str, extension:str=".md") -> str:
    """
    This function will create a filename from a topic (unformatted string) and an extension.
    Args:
      topic: str
      extension: str
    Returns:
      str: The generated filename with the specified extension
    """
    filename = slugify(topic, separator='_')
    return filename + extension

# Function to detect language from file extension
def detect_language(filename):
        """Detect programming language from file extension"""
        ext = os.path.splitext(filename)[1].lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.php': 'php',
            '.pas': 'pascal',
            '.pp': 'pascal',
            '.lpr': 'pascal',
            '.dpr': 'pascal',
            '.inc': 'pascal',
            '.md': 'markdown',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'cpp',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.m': 'objective-c',
            '.scala': 'scala',
            '.sh': 'bash',
            '.bash': 'bash',
            '.zsh': 'bash',
            '.r': 'r',
            '.lua': 'lua',
            '.dart': 'dart',
            '.html': 'html',
            '.css': 'css',
            '.xml': 'xml',
            '.json': 'json',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.pl': 'perl',
            '.cob': 'cobol',
            '.cbl': 'cobol',
            '.sql': 'sql',
            '.vb': 'vbnet',
            '.vbnet': 'vbnet',
            '.erl': 'erlang',
            '.ex': 'elixir',
            '.exs': 'elixir',
            '.hs': 'haskell',
            '.jl': 'julia',
            '.groovy': 'groovy',
            '.ps1': 'powershell',
            '.psm1': 'powershell',
        }
        return language_map.get(ext, 'generic')

@tool
def list_directory_tree(folder_path: str, max_depth: int = 6, show_files: bool = True, add_function_signatures: bool = False) -> str:
    """
    Creates a tree-like view of a directory structure. This is useful for understanding
    project structure without loading all file contents, saving context.
    
    Example output:
    project/
    ├── src/
    │   ├── main.py (123 lines)
    │   └── utils.py (45 lines)
    └── tests/
        └── test_main.py (67 lines)
    
    Total source code lines: 235
    
    Args:
        folder_path: str The root folder path to visualize
        max_depth: int Maximum depth to traverse (default 3)
        show_files: bool Whether to show files or only directories (default True)
        add_function_signatures: bool Whether to extract and display function signatures for source code files (default False)
    
    Returns:
        str: A string representation of the directory tree
    """
    if not os.path.isdir(folder_path):
        return f"Error: '{folder_path}' is not a valid directory"

    lines = []
    total_lines = 0

    def add_tree_lines(current_path, prefix="", depth=0):
        nonlocal total_lines
        if depth > max_depth:
            return

        try:
            items = sorted(os.listdir(current_path))
        except PermissionError:
            return

        # Filter out hidden files/folders starting with '.'
        items = [item for item in items if not item.startswith('.')]

        # Separate directories and files
        dirs = [item for item in items if os.path.isdir(os.path.join(current_path, item))]
        files = [item for item in items if os.path.isfile(os.path.join(current_path, item))] if show_files else []

        all_items = dirs + files

        for i, item in enumerate(all_items):
            is_last = i == len(all_items) - 1
            item_path = os.path.join(current_path, item)

            # Choose the appropriate tree characters
            connector = "└── " if is_last else "├── "

            # Check if file is a source code file and count lines
            line_count_str = ""
            if os.path.isfile(item_path):
                _, ext = os.path.splitext(item)
                if ext.lower() in DEFAULT_SOURCE_CODE_EXTENSIONS:
                    try:
                        num_lines = get_file_lines(item_path)
                        line_word = "line" if num_lines == 1 else "lines"
                        line_count_str = f" ({num_lines} {line_word})"
                        total_lines += num_lines
                    except (UnicodeDecodeError, PermissionError, OSError, IOError):
                        # If we can't read the file, just skip the line count
                        pass

            lines.append(f"{prefix}{connector}{item}{'/' if os.path.isdir(item_path) else ''}{line_count_str}")

            # Extract and display function signatures if requested
            if add_function_signatures and os.path.isfile(item_path):
                _, ext = os.path.splitext(item)
                if ext.lower() in DEFAULT_SOURCE_CODE_EXTENSIONS:
                    try:
                        language = detect_language(item)
                        signatures = extract_function_signatures(item_path, language)
                        
                        # Only display if signatures were found and don't contain error messages
                        # Check if the result contains actual signatures (not error or empty messages)
                        if (
                            signatures
                            and not signatures.startswith("Error:")
                            and not signatures.startswith("No function")
                            and not signatures.startswith("No sections")
                        ):
                            # Indent the signatures to align with the tree structure
                            extension = "    " if is_last else "│   "
                            sig_prefix = prefix + extension
                            for sig in signatures.split('\n'):
                                if sig.strip():  # Only add non-empty signature lines
                                    lines.append(f"{sig_prefix}    {sig.strip()}")
                    except (
                        UnicodeDecodeError,
                        PermissionError,
                        OSError,
                        IOError,
                    ):
                        # Silently skip files that cause errors during signature extraction
                        pass

            # Recurse into subdirectories
            if os.path.isdir(item_path):
                extension = "    " if is_last else "│   "
                add_tree_lines(item_path, prefix + extension, depth + 1)

    # Add root folder
    lines.append(f"{os.path.basename(folder_path)}/")
    add_tree_lines(folder_path, "", 0)

    # Add total line count at the end if any source code files were found
    if total_lines > 0:
        lines.append(f"\nTotal source code lines: {total_lines}")

    return "\n".join(lines)

@tool
def search_in_files(folder_path: str, search_pattern: str, file_extensions: tuple = None, 
                    case_sensitive: bool = False, max_results: int = 50) -> str:
    """
    Searches for a pattern in files within a folder and its subfolders.
    Returns matching lines with file paths and line numbers. This is much more efficient
    than loading all files when you need to find specific code patterns.
    
    Args:
        folder_path: str The root folder to search in
        search_pattern: str The text pattern to search for
        file_extensions: tuple Optional tuple of file extensions to search (e.g., ('.py', '.js'))
                        If None, searches all text files
        case_sensitive: bool Whether the search should be case-sensitive (default False)
        max_results: int Maximum number of results to return (default 50)
    
    Returns:
        str: Search results formatted as "filepath:line_number: line_content"
    """
    if not os.path.isdir(folder_path):
        return f"Error: '{folder_path}' is not a valid directory"

    results = []
    count = 0

    # Prepare search pattern
    pattern = search_pattern if case_sensitive else search_pattern.lower()

    for root, _, files in os.walk(folder_path):
        # Skip hidden directories
        if any(part.startswith('.') for part in root.split(os.sep)):
            continue

        for filename in files:
            # Skip hidden files
            if filename.startswith('.'):
                continue

            # Filter by extension if specified
            if file_extensions:
                if not any(filename.endswith(ext) for ext in file_extensions):
                    continue

            filepath = os.path.join(root, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        search_line = line if case_sensitive else line.lower()
                        if pattern in search_line:
                            results.append(f"{filepath}:{line_num}: {line.rstrip()}")
                            count += 1
                            if count >= max_results:
                                results.append(f"\n... (stopped at {max_results} results)")
                                return "\n".join(results)
            except (UnicodeDecodeError, PermissionError, IOError):
                # Skip files that can't be read
                continue

    if not results:
        return f"No matches found for '{search_pattern}' in '{folder_path}'"

    return "\n".join(results)

@tool
def read_file_range(filename: str, start_byte: int, end_byte: int) -> str:
    """
    Reads a specific byte range from a file. This is useful for very large files
    where you only need to inspect a portion, saving memory and context.
    
    Args:
        filename: str The file path
        start_byte: int The starting byte position (0-indexed)
        end_byte: int The ending byte position (exclusive)
    
    Returns:
        str: The content from the specified byte range
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' not found")

    if start_byte < 0 or end_byte < 0:
        raise ValueError("Byte positions must be non-negative")

    if start_byte >= end_byte:
        raise ValueError("start_byte must be less than end_byte")

    try:
        with open(filename, 'rb') as f:
            f.seek(start_byte)
            bytes_to_read = end_byte - start_byte
            content = f.read(bytes_to_read)
            return content.decode('utf-8', errors='replace')
    except Exception as e:
        raise IOError(f"Error reading file range: {e}")

@tool
def get_file_info(filepath: str) -> dict:
    """
    Gets metadata about a file without reading its content. This is efficient
    for checking file properties before deciding whether to load the full content.
    
    Args:
        filepath: str The file path
    
    Returns:
        dict: Dictionary containing file metadata (size, modified_time, is_file, is_dir, exists)
    """
    info = {
        'exists': os.path.exists(filepath),
        'is_file': os.path.isfile(filepath),
        'is_dir': os.path.isdir(filepath),
        'size_bytes': 0,
        'modified_time': None,
        'readable': False,
        'writable': False
    }

    if info['exists']:
        try:
            stat_info = os.stat(filepath)
            info['size_bytes'] = stat_info.st_size
            info['modified_time'] = stat_info.st_mtime
            info['readable'] = os.access(filepath, os.R_OK)
            info['writable'] = os.access(filepath, os.W_OK)
        except (PermissionError, OSError):
            pass

    return info

@tool
def list_directory(folder_path: str, pattern: str = "*", recursive: bool = False, 
                   files_only: bool = False, dirs_only: bool = False) -> list:
    """
    Lists files and directories in a folder with optional filtering.
    More flexible than get_files_in_folder with pattern matching support.
    
    Args:
        folder_path: str The folder path to list
        pattern: str Glob pattern to match (default "*" for all)
        recursive: bool Whether to search recursively (default False)
        files_only: bool Return only files (default False)
        dirs_only: bool Return only directories (default False)
    
    Returns:
        list: List of matching paths
    """
    if not os.path.isdir(folder_path):
        return []

    import glob

    if recursive:
        search_pattern = os.path.join(folder_path, "**", pattern)
        matches = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(folder_path, pattern)
        matches = glob.glob(search_pattern)

    # Filter based on type
    if files_only:
        matches = [m for m in matches if os.path.isfile(m)]
    elif dirs_only:
        matches = [m for m in matches if os.path.isdir(m)]

    return sorted(matches)

@tool
def mkdir(directory_path: str, parents: bool = True) -> bool:
    """
    Creates a directory. If parents=True, creates intermediate directories as needed.
    
    Args:
        directory_path: str The directory path to create
        parents: bool Whether to create parent directories (default True)
    
    Returns:
        bool: True if successful, raises exception otherwise
    """
    try:
        if parents:
            os.makedirs(directory_path, exist_ok=True)
        else:
            os.mkdir(directory_path)
        return True
    except Exception as e:
        raise OSError(f"Failed to create directory '{directory_path}': {e}")

@tool
def extract_function_signatures(filename: str, language: str = "python") -> str:
    """
    Extracts function and class signatures from a source code file without loading
    the full implementation. This helps understand code structure efficiently.

    Currently supports: python, javascript, java, php, pascal, and generic fallback for most languages

    Args:
        filename: str The source code file path
        language: str The programming language (default "python")

    Returns:
        str: Extracted signatures, one per line
    """
    if not os.path.isfile(filename):
        return f"Error: File '{filename}' not found"

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    signatures = []

    if language.lower() in ["markdown", "md"]:
        # Extract markdown sections (lines starting with #)
        # Skip lines inside code blocks (fenced with ``` or ~~~)
        md_sections = []
        code_block_delimiter = None  # Stores the opening delimiter type
        for line in content.split('\n'):
            stripped = line.strip()
            # Check for code block delimiters (``` or ~~~)
            if code_block_delimiter is None:
                # Not in a code block - check for opening delimiter
                if stripped.startswith('```'):
                    code_block_delimiter = '```'
                    continue
                elif stripped.startswith('~~~'):
                    code_block_delimiter = '~~~'
                    continue
            else:
                # Inside a code block - check for matching closing delimiter
                if stripped.startswith(code_block_delimiter):
                    code_block_delimiter = None
                    continue
            # Only extract headers when not inside a code block
            if code_block_delimiter is None and stripped.startswith('#'):
                md_sections.append(stripped)
        if not md_sections:
            return f"No sections found in '{filename}'"
        return "\n".join(md_sections)

    elif language.lower() == "python":
        # Match Python function and class definitions
        # Match def function_name(...): and class ClassName(...):
        pattern = r'^([ \t]*)(def|class)\s+(\w+)\s*(\([^)]*\))?\s*:'
        for match in re.finditer(pattern, content, re.MULTILINE):
            indent = match.group(1)
            keyword = match.group(2)
            name = match.group(3)
            params = match.group(4) or ""
            signatures.append(f"{indent}{keyword} {name}{params}:")

    elif language.lower() in ["javascript", "js", "typescript", "ts"]:
        # Match JavaScript/TypeScript function declarations
        # function name(...), async function name(...), name(...) {, const name = (...) =>
        patterns = [
            r'^([ \t]*)(async\s+)?function\s+(\w+)\s*(\([^)]*\))',
            r'^([ \t]*)(\w+)\s*(\([^)]*\))\s*\{',
            r'^([ \t]*)(const|let|var)\s+(\w+)\s*=\s*(\([^)]*\))\s*=>'
        ]
        matches_with_pos = []
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                sig = match.group(0).split('{')[0].strip()
                matches_with_pos.append((match.start(), sig))
        # Sort by position in source code and deduplicate
        matches_with_pos.sort(key=lambda x: x[0])
        seen = set()
        for _, sig in matches_with_pos:
            if sig not in seen:
                signatures.append(sig)
                seen.add(sig)

    elif language.lower() == "java":
        # Match Java class, interface, enum declarations and method declarations
        patterns = [
            r'^([ \t]*)(public|private|protected)?\s*(abstract|final)?\s*(class|interface|enum)\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{?',
            r'^([ \t]*)(public|private|protected)?\s*(static)?\s*(\w+)\s+(\w+)\s*(\([^)]*\))'
        ]
        matches_with_pos = []
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                sig = match.group(0).strip()
                matches_with_pos.append((match.start(), sig))
        # Sort by position in source code and deduplicate
        matches_with_pos.sort(key=lambda x: x[0])
        seen = set()
        for _, sig in matches_with_pos:
            if sig not in seen:
                signatures.append(sig)
                seen.add(sig)

    elif language.lower() == "php":
        # Match PHP function and method declarations (including object-oriented features)
        # Matches: function name(...), class ClassName, public/private/protected function name(...)
        patterns = [
            r'^([ \t]*)class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{?',
            r'^([ \t]*)(public|private|protected)\s+(static\s+)?function\s+(\w+)\s*(\([^)]*\))',
            r'^([ \t]*)function\s+(\w+)\s*(\([^)]*\))'
        ]
        matches_with_pos = []
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                sig = match.group(0).strip()
                matches_with_pos.append((match.start(), sig))
        # Sort by position in source code and deduplicate
        matches_with_pos.sort(key=lambda x: x[0])
        seen = set()
        for _, sig in matches_with_pos:
            if sig not in seen:
                signatures.append(sig)
                seen.add(sig)

    elif language.lower() in ["pascal", "objectpascal", "delphi"]:
        # Match Pascal/Object Pascal function, procedure, class, record, and interface declarations
        # Handles: function Name(...): Type; procedure Name(...); TClassName = class; TRecord = record; IInterface = interface
        patterns = [
            r'^([ \t]*)(function|procedure)\s+(\w+)\s*(\([^)]*\))?(?:\s*:\s*\w+)?\s*;',
            r'^([ \t]*)(class\s+)?(function|procedure)\s+(\w+)\s*(\([^)]*\))?(?:\s*:\s*\w+)?\s*;',
            r'^([ \t]*)(type\s+)?(\w+)\s*=\s*(class|record|interface|object)(?:\s*\([^)]*\))?',
            r'^([ \t]*)(constructor|destructor)\s+(\w+)\s*(\([^)]*\))?\s*;?'
        ]
        matches_with_pos = []
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                sig = match.group(0).strip()
                matches_with_pos.append((match.start(), sig))
        # Sort by position in source code and deduplicate
        matches_with_pos.sort(key=lambda x: x[0])
        seen = set()
        for _, sig in matches_with_pos:
            if sig not in seen:
                signatures.append(sig)
                seen.add(sig)

    elif language.lower() in ["c", "cpp", "c++", "cxx", "h", "hpp"]:
        # Match C/C++ struct, class (C++), and function declarations
        # Use explicit list of common C/C++ return types for better precision
        c_types = r'(?:void|int|char|short|long|float|double|unsigned|signed|bool|size_t|ssize_t|auto|const\s+\w+|\w+_t)'
        patterns = [
            r'^([ \t]*)(typedef\s+)?struct\s+(\w+)?\s*\{?',
            r'^([ \t]*)class\s+(\w+)(?:\s*:\s*(public|private|protected)?\s*\w+)?\s*\{?',
            r'^([ \t]*)' + c_types + r'(?:\s*\*)*\s+(\w+)\s*(\([^)]*\))\s*[{;]?'
        ]
        matches_with_pos = []
        for pattern in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                sig = match.group(0).strip()
                matches_with_pos.append((match.start(), sig))
        # Sort by position in source code and deduplicate
        matches_with_pos.sort(key=lambda x: x[0])
        seen = set()
        for _, sig in matches_with_pos:
            if sig not in seen:
                signatures.append(sig)
                seen.add(sig)

    else:
        # Generic fallback for unsupported languages using "function" and "procedure" keywords
        # This will work for many programming languages that use these keywords
        # Conservative pattern that matches common function/procedure declarations
        pattern = r'^([ \t]*)(function|procedure)\s+(\w+)\s*(\([^)]*\))?'
        seen = set()
        for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
            sig = match.group(0).strip()
            if sig not in seen:
                signatures.append(sig)
                seen.add(sig)

    if not signatures:
        return f"No function/class signatures found in '{filename}'"

    return "\n".join(signatures)

@tool
def compare_files(file1: str, file2: str, context_lines: int = 3) -> str:
    """
    Compares two files and shows the differences in a unified diff format.
    Useful for understanding what changed between versions.
    
    Args:
        file1: str Path to the first file
        file2: str Path to the second file
        context_lines: int Number of context lines to show around differences (default 3)
    
    Returns:
        str: Unified diff output
    """
    if not os.path.isfile(file1):
        return f"Error: File '{file1}' not found"
    if not os.path.isfile(file2):
        return f"Error: File '{file2}' not found"

    try:
        with open(file1, 'r', encoding='utf-8') as f:
            lines1 = f.readlines()
        with open(file2, 'r', encoding='utf-8') as f:
            lines2 = f.readlines()
    except Exception as e:
        return f"Error reading files: {e}"

    diff = difflib.unified_diff(
        lines1, lines2,
        fromfile=file1,
        tofile=file2,
        lineterm='',
        n=context_lines
    )

    diff_output = '\n'.join(diff)

    if not diff_output:
        return "Files are identical"

    return diff_output

@tool
def compare_folders(folder1: str, folder2: str, context_lines: int = 3) -> str:
    """
    Compares two folders and shows the differences for source code files.
    Only files with extensions in DEFAULT_SOURCE_CODE_EXTENSIONS are compared.
    
    Args:
        folder1: str Path to the first folder
        folder2: str Path to the second folder
        context_lines: int Number of context lines to show around differences (default 3)
    
    Returns:
        str: Comparison report showing files only in each folder and diffs for changed files
    """
    if not os.path.isdir(folder1):
        return f"Error: Folder '{folder1}' not found"
    if not os.path.isdir(folder2):
        return f"Error: Folder '{folder2}' not found"
    
    # Cache lowercased extensions for performance
    source_extensions = tuple(ext.lower() for ext in DEFAULT_SOURCE_CODE_EXTENSIONS)
    
    # Get all source code files from both folders
    def get_source_files(folder):
        """Get all source code files recursively from a folder"""
        source_files = {}
        for root, _, files in os.walk(folder):
            for filename in files:
                # Check if file has a source code extension
                if filename.lower().endswith(source_extensions):
                    full_path = os.path.join(root, filename)
                    # Store relative path as key
                    rel_path = os.path.relpath(full_path, folder)
                    source_files[rel_path] = full_path
        return source_files
    
    files1 = get_source_files(folder1)
    files2 = get_source_files(folder2)
    
    # Find files only in folder1, only in folder2, and in both
    only_in_folder1 = set(files1.keys()) - set(files2.keys())
    only_in_folder2 = set(files2.keys()) - set(files1.keys())
    common_files = set(files1.keys()) & set(files2.keys())
    
    # Build the comparison report
    output = []
    
    # Summary
    output.append("=== FOLDER COMPARISON SUMMARY ===")
    output.append(f"Folder 1: {folder1}")
    output.append(f"Folder 2: {folder2}")
    output.append(f"Files only in folder 1: {len(only_in_folder1)}")
    output.append(f"Files only in folder 2: {len(only_in_folder2)}")
    output.append(f"Common files: {len(common_files)}")
    output.append("")
    
    # Files only in folder1
    if only_in_folder1:
        output.append("=== FILES ONLY IN FOLDER 1 ===")
        for file in sorted(only_in_folder1):
            output.append(f"  {file}")
        output.append("")
    
    # Files only in folder2
    if only_in_folder2:
        output.append("=== FILES ONLY IN FOLDER 2 ===")
        for file in sorted(only_in_folder2):
            output.append(f"  {file}")
        output.append("")
    
    # Compare common files
    different_files = []
    identical_files = []
    
    for file in sorted(common_files):
        path1 = files1[file]
        path2 = files2[file]
        
        try:
            # Try utf-8 first, then fallback to latin-1 like load_string_from_file
            try:
                with open(path1, 'r', encoding='utf-8') as f:
                    lines1 = f.readlines()
            except UnicodeDecodeError:
                with open(path1, 'r', encoding='latin-1') as f:
                    lines1 = f.readlines()
            
            try:
                with open(path2, 'r', encoding='utf-8') as f:
                    lines2 = f.readlines()
            except UnicodeDecodeError:
                with open(path2, 'r', encoding='latin-1') as f:
                    lines2 = f.readlines()
            
            # Check if files are different
            if lines1 != lines2:
                different_files.append((file, path1, path2, lines1, lines2))
            else:
                identical_files.append(file)
        except Exception as e:
            output.append(f"Error comparing {file}: {e}")
            output.append("")
    
    # Report identical and different files
    output.append(f"=== COMPARISON RESULTS ===")
    output.append(f"Identical files: {len(identical_files)}")
    output.append(f"Different files: {len(different_files)}")
    output.append("")
    
    # Show diffs for different files
    if different_files:
        output.append("=== DIFFERENCES ===")
        for file, path1, path2, lines1, lines2 in different_files:
            output.append(f"\n--- {file} ---")
            diff = difflib.unified_diff(
                lines1, lines2,
                fromfile=f"folder1/{file}",
                tofile=f"folder2/{file}",
                lineterm='',
                n=context_lines
            )
            diff_output = '\n'.join(diff)
            output.append(diff_output)
            output.append("")
    
    # If folders are identical
    if not only_in_folder1 and not only_in_folder2 and not different_files:
        return "Folders are identical (all source code files match)"
    
    return '\n'.join(output)

@tool
def delete_file(filepath: str) -> bool:
    """
    Deletes a file from the filesystem.
    
    Args:
        filepath: str Path to the file to delete
    
    Returns:
        bool: True if successful
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found")

    if os.path.isdir(filepath):
        raise IsADirectoryError(f"'{filepath}' is a directory, use delete_directory instead")

    try:
        os.remove(filepath)
        return True
    except Exception as e:
        raise OSError(f"Failed to delete file '{filepath}': {e}")

@tool
def delete_directory(directory_path: str, recursive: bool = False) -> bool:
    """
    Deletes a directory. If recursive=True, deletes the directory and all its contents.
    
    Args:
        directory_path: str Path to the directory to delete
        recursive: bool Whether to delete recursively (default False)
    
    Returns:
        bool: True if successful
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' not found")

    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"'{directory_path}' is not a directory")

    try:
        if recursive:
            import shutil
            shutil.rmtree(directory_path)
        else:
            os.rmdir(directory_path)
        return True
    except Exception as e:
        raise OSError(f"Failed to delete directory '{directory_path}': {e}")

@tool
def count_lines_of_code(folder_path: str, file_extensions: tuple = ('.py', '.js', '.java', '.cpp', '.c', '.php', '.rb')) -> dict:
    """
    Counts lines of code in a project, broken down by file type.
    Helps understand project size and composition without loading all files.
    
    Args:
        folder_path: str Root folder to analyze
        file_extensions: tuple File extensions to count (default common programming languages)
    
    Returns:
        dict: Dictionary with file extension as key and line count as value
    """
    if not os.path.isdir(folder_path):
        return {"error": f"'{folder_path}' is not a valid directory"}

    counts = {}
    total_lines = 0

    for root, _, files in os.walk(folder_path):
        # Skip hidden directories
        if any(part.startswith('.') for part in root.split(os.sep)):
            continue

        for filename in files:
            if filename.startswith('.'):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext in file_extensions:
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                        counts[ext] = counts.get(ext, 0) + line_count
                        total_lines += line_count
                except (UnicodeDecodeError, PermissionError, IOError):
                    continue

    counts['_total'] = total_lines
    return counts

@tool
def read_first_n_lines(filename: str, n: int) -> str:
    """
    Reads the first n lines of a file. Useful for previewing large files
    without loading everything into memory.

    Args:
        filename: str Path to the file
        n: int Number of lines to read

    Returns:
        str: The first n lines of the file

    Example:
        # Print first 50 lines
        print(read_first_n_lines('code.py', 50))
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' not found")

    cnt_lines = get_file_lines(filename)

    if n < 1:
        raise ValueError("n must be >= 1")

    if  n > cnt_lines:
        n = cnt_lines

    lines = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                lines.append(line)
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='latin-1') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                lines.append(line)

    return ''.join(lines)

@tool
def read_last_n_lines(filename: str, n: int) -> str:
    """
    Reads the last n lines of a file. Useful for reading log files or
    checking the end of large files.

    Args:
        filename: str Path to the file
        n: int Number of lines to read from the end

    Returns:
        str: The last n lines of the file
    
    Example:
        # Print last 50 lines
        print(read_last_n_lines('code.py', 50))
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' not found")
    cnt_lines = get_file_lines(filename)

    if n < 1:
        raise ValueError("n must be >= 1")

    if  n > cnt_lines:
        n = cnt_lines

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    return ''.join(lines[-n:])

@tool
def delete_lines_from_file(filename: str, start_line: int, end_line: int = None) -> str:
    """
    Deletes specific lines from a file.

    Args:
        filename: str Path to the file
        start_line: int The first line to delete (1-based index)
        end_line: int The last line to delete (1-based, inclusive).
                  If None, only deletes the start_line.

    Returns:
        str: The updated file content

    Example:
        # Delete line 5
        delete_lines_from_file('code.py', 5)

        # Delete lines 10-15 (inclusive)
        delete_lines_from_file('code.py', 10, 15)
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' not found")

    cnt_lines = get_file_lines(filename)

    if start_line < 1:
        raise ValueError("start_line must be >= 1")

    if end_line is None:
        end_line = start_line

    if end_line > cnt_lines:
        end_line = cnt_lines

    if end_line < start_line:
        raise ValueError("end_line must be >= start_line")

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    if start_line > total_lines:
        raise IndexError(f"start_line {start_line} is beyond file length ({total_lines} lines)")

    # Remove the specified lines (convert to 0-based index)
    del lines[start_line - 1:end_line]

    content = ''.join(lines)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

    return content


class PlanningTool(Tool):
    """A tool that lets the agent trigger a planning step on demand.

    The tool makes a single LLM call (no tools, no agent loop) to generate or
    update a plan based on the current task, memory, and available capabilities.
    It must be bound to an agent via ``set_agent`` before use.
    """

    name = "plan"
    description = (
        "Call this tool whenever you need help to create or update your plan. "
        "Use it when starting a complex task, when your current approach is failing, "
        "when the task scope changes, or when you feel stuck. "
        "Provide a short summary of what happened so far and what you need to plan for."
        "I do have access to your recent steps."
    )
    inputs = {
        "situation": {
            "type": "string",
            "description": (
                "A brief description of the current situation: what has been tried, "
                "what worked, what failed, and what needs to be planned next."
            ),
        }
    }
    output_type = "string"

    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        self._agent = None
        self._planning_model = model

    def set_agent(self, agent):
        """Bind this tool to an agent so it can access task, memory, tools, and model."""
        self._agent = agent

    def forward(self, situation: str) -> str:
        if self._agent is None:
            return "Error: PlanningTool is not bound to an agent. Call set_agent() first."

        agent = self._agent
        model = self._planning_model or agent.model
        task = agent.task or "No task set"
        remaining_steps = agent.max_steps - agent.step_number

        # Build tool/agent capability list
        capabilities = []
        for t in agent.tools.values():
            if t.name != self.name:  # Don't list ourselves
                capabilities.append(f"- {t.name}: {t.description}")
        if hasattr(agent, "managed_agents") and agent.managed_agents:
            for a in agent.managed_agents.values():
                capabilities.append(f"- {a.name} (team member): {a.description}")
        capabilities_text = "\n".join(capabilities) if capabilities else "No tools available."

        # Build memory summary (last steps only, to keep prompt small)
        memory_summary = ""
        if agent.memory and agent.memory.steps:
            recent = agent.memory.steps[-22:]
            parts = []
            for step in recent:
                messages = step.to_messages(summary_mode=True)
                for msg in messages:
                    if isinstance(msg.content, str):
                        parts.append(msg.content[:5000])
                    elif isinstance(msg.content, list):
                        for block in msg.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                parts.append(block["text"][:5000])
            memory_summary = "\n---\n".join(parts)

        prompt = textwrap.dedent(f"""\
            You are a planning assistant. Your ONLY job is to produce a clear,actionable plan.
            Do NOT write code. Do NOT execute actions. Just plan.

            ## Task
            {task}

            ## Current situation
            {situation}

            ## Recent history
            {memory_summary if memory_summary else "No history yet."}

            ## Available capabilities
            {capabilities_text}

            ## Constraints
            - {remaining_steps} steps remaining.

            ## Instructions
            Write a step-by-step plan to solve the task.
            Mark completed steps with [X] and pending steps with [ ].
            Try to make the smallest possible plan to achieve the goal keeping good outcome quality.
            Be concise and actionable. End with <end_plan>.
        """)

        input_messages = [
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": prompt}],
            )
        ]

        response = model.generate(input_messages, stop_sequences=["<end_plan>"])
        plan_text = response.content if isinstance(response.content, str) else str(response.content)

        # Reset the planning interval so the next scheduled plan counts from this step
        agent._last_plan_step = agent.step_number

        return plan_text