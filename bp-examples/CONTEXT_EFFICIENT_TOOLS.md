# Context-Efficient Tools for AI Agents

This document describes the new context-efficient tools added to Beyond Python Smolagents. These tools are designed to help AI agents work with codebases efficiently without burning their context window.

## The Problem

When AI agents work with large codebases, they often face context limitations. Loading all files into context is:
- **Wasteful**: Most of the loaded content may not be relevant
- **Inefficient**: Large context windows are slower and more expensive
- **Limiting**: May hit context limits before solving the problem

## The Solution

We've added a suite of tools that allow agents to:
1. **Explore before loading**: Understand structure without loading content
2. **Search efficiently**: Find relevant code without loading all files
3. **Load strategically**: Make informed decisions about what to load
4. **Work incrementally**: Process large files in chunks

## New Tools Overview

### 🗂️ Project Structure Tools

#### `list_directory_tree(folder_path, max_depth=3, show_files=True)`
Creates a tree-like view of directory structure.

**Example:**
```python
tree = list_directory_tree('src/smolagents', max_depth=2)
print(tree)
# Output:
# smolagents/
# ├── agents.py
# ├── tools.py
# └── models.py
```

**Use Case**: Get project overview before diving into specific files.

---

#### `list_directory(folder_path, pattern="*", recursive=False, files_only=False, dirs_only=False)`
Lists files and directories with glob pattern matching.

**Example:**
```python
# Find all Python test files
test_files = list_directory('tests', pattern='test_*.py', recursive=True, files_only=True)

# Find all directories
dirs = list_directory('src', dirs_only=True)
```

**Use Case**: Find specific types of files without loading them.

---

### 🔍 Code Search Tools

#### `search_in_files(folder_path, search_pattern, file_extensions=None, case_sensitive=False, max_results=50)`
Searches for patterns in files, returns matches with line numbers.

**Example:**
```python
# Find all function definitions
results = search_in_files(
    'src', 
    'def ', 
    file_extensions=('.py',),
    max_results=20
)
# Output: filepath:line_number: line_content
```

**Use Case**: Find where specific code patterns exist without loading all files.

---

#### `extract_function_signatures(filename, language="python")`
Extracts function and class signatures without implementations.

**Example:**
```python
signatures = extract_function_signatures('src/tools.py', 'python')
# Output:
# def process_data(input: str):
# class DataProcessor:
# def validate(self, data):
```

**Supported Languages**: Python, JavaScript, TypeScript, Java, PHP

**Use Case**: Understand API surface without loading full implementations.

---

### 📊 File Information Tools

#### `get_file_info(filepath)`
Gets file metadata without reading content.

**Example:**
```python
info = get_file_info('large_file.txt')
print(f"Size: {info['size_bytes']} bytes")
print(f"Modified: {info['modified_time']}")
print(f"Readable: {info['readable']}")
```

**Use Case**: Check file size before deciding to load it. Avoid accidentally loading huge files.

---

#### `count_lines_of_code(folder_path, file_extensions=('.py', '.js', ...))`
Counts lines of code by file type.

**Example:**
```python
stats = count_lines_of_code('src', file_extensions=('.py', '.js'))
# Output: {'.py': 5000, '.js': 2000, '_total': 7000}
```

**Use Case**: Understand project scale before diving in.

---

### 📖 Efficient File Reading

#### `read_file_range(filename, start_byte, end_byte)`
Reads specific byte range from a file.

**Example:**
```python
# Read first 1000 bytes of a large log file
content = read_file_range('huge_log.txt', 0, 1000)

# Read middle section
content = read_file_range('data.csv', 10000, 20000)
```

**Use Case**: Inspect parts of very large files without loading the entire file.

---

### 🔄 File Comparison

#### `compare_files(file1, file2, context_lines=3)`
Shows differences between two files in unified diff format.

**Example:**
```python
diff = compare_files('original.py', 'modified.py', context_lines=3)
print(diff)
# Shows unified diff with +/- lines
```

**Use Case**: Understand what changed between versions before deciding what to do.

---

### 🛠️ File System Operations

#### `mkdir(directory_path, parents=True)`
Creates directories, optionally creating parent directories.

**Example:**
```python
mkdir('output/results/data', parents=True)
```

---

#### `delete_file(filepath)`
Deletes a file.

**Example:**
```python
delete_file('temp_output.txt')
```

---

#### `delete_directory(directory_path, recursive=False)`
Deletes a directory, optionally with all contents.

**Example:**
```python
delete_directory('temp_dir', recursive=True)
```

---

## Recommended Workflow for Agents

Here's an efficient workflow that saves context:

### 1️⃣ Survey Phase (Minimal Context Usage)
```python
# Get project overview
tree = list_directory_tree('.', max_depth=3)

# Understand scale
metrics = count_lines_of_code('.', ('.py', '.js'))

# Find relevant files
relevant = search_in_files('.', 'class MyClass', file_extensions=('.py',))
```

### 2️⃣ Investigation Phase (Strategic Loading)
```python
# Check file sizes before loading
for file in candidate_files:
    info = get_file_info(file)
    if info['size_bytes'] < 100000:  # Only load files < 100KB
        content = load_string_from_file(file)
        # Process content...
    else:
        # Use extract_function_signatures or read_file_range for large files
        signatures = extract_function_signatures(file)
```

### 3️⃣ Implementation Phase (Focused Changes)
```python
# Load only what you need
target_file = load_string_from_file('src/target.py')

# Make changes
replace_on_file('src/target.py', old_code, new_code)

# Verify changes
diff = compare_files('src/target.py.bak', 'src/target.py')
```

## Context Savings Example

### ❌ Inefficient Approach
```python
# Load everything (wastes context!)
all_code = source_code_to_string('large_project')  # 50,000 lines
# Now try to find what you need in 50,000 lines...
```

### ✅ Efficient Approach
```python
# 1. Understand structure (100 lines context)
tree = list_directory_tree('large_project', max_depth=3)

# 2. Find relevant files (200 lines context)
matches = search_in_files('large_project', 'MyTargetClass')

# 3. Load only relevant file (500 lines context)
target_file = load_string_from_file('src/relevant.py')

# Total: ~800 lines vs 50,000 lines (98.4% context savings!)
```

## Integration with Default Tools

These tools are automatically included in `DEFAULT_THINKER_TOOLS`:
- `list_directory_tree`
- `search_in_files`
- `get_file_info`
- `list_directory`
- `extract_function_signatures`
- `compare_files`
- `count_lines_of_code`
- `mkdir`
- `delete_file`
- `delete_directory`

## Usage in Agent Code

```python
from smolagents import CodeAgent, LiteLLMModel
from smolagents.bp_tools import *
from smolagents.bp_thinkers import DEFAULT_THINKER_TOOLS

# These tools are already in DEFAULT_THINKER_TOOLS
model = LiteLLMModel(model_id="gemini/gemini-2.5-flash", api_key="your-key")
agent = CodeAgent(model=model, tools=DEFAULT_THINKER_TOOLS, add_base_tools=True)

task = """
Analyze the structure of this project and find all classes that inherit from 'Tool'.
Use context-efficient tools to avoid loading everything.
"""

agent.run(task)
```

## Best Practices

### ✅ DO:
- Use `list_directory_tree` first to understand structure
- Use `search_in_files` to find relevant code
- Check `get_file_info` before loading large files
- Use `extract_function_signatures` to understand APIs
- Load full files only when necessary

### ❌ DON'T:
- Load all files at once with `source_code_to_string` on large projects
- Skip the exploration phase
- Load files without checking their size first
- Use `load_string_from_file` on very large files without inspection

## Examples

See `bp-examples/context-efficient-tools-demo.py` for a comprehensive demonstration of all tools.

Run the demo:
```bash
python bp-examples/context-efficient-tools-demo.py
```

## Summary

These context-efficient tools enable AI agents to:
- 📊 **Survey** projects efficiently
- 🎯 **Search** for relevant code precisely  
- 💡 **Decide** what to load strategically
- ⚡ **Work** within context limits effectively

By using these tools, agents can handle much larger codebases while staying within context windows and working more efficiently.
