# Example: Agent Using Context-Efficient Tools

This document shows an example of how an AI agent would use the new context-efficient tools to analyze and modify a codebase efficiently.

## Scenario

An agent is asked: **"Find all Python files in the project that define classes, and add type hints to their `__init__` methods."**

## Traditional Approach (Inefficient) ❌

```python
# Agent loads ALL files at once - wastes context!
all_code = source_code_to_string('.')  # Could be 100,000+ lines!

# Now agent has to search through all this loaded content
# Most of it is irrelevant (tests, docs, config files, etc.)
```

**Problems:**
- Loads 100,000+ lines into context
- Wastes most of the context on irrelevant files
- May hit context limits before completing the task
- Slower and more expensive

## New Approach (Efficient) ✅

### Step 1: Survey the Project

```python
# Get an overview of the project structure
tree = list_directory_tree('.', max_depth=3, show_files=True)
print(tree)
```

**Output (example):**
```
project/
├── src/
│   ├── models.py
│   ├── utils.py
│   └── config.py
├── tests/
│   └── test_models.py
└── docs/
    └── README.md
```

**Agent reasoning:** "I can see there's a `src/` directory with the main code and a `tests/` directory. I should focus on `src/`."

**Context used:** ~50 lines

---

### Step 2: Find Relevant Files

```python
# Search for class definitions in Python files
results = search_in_files(
    'src',
    'class ',
    file_extensions=('.py',),
    max_results=50
)
print(results)
```

**Output (example):**
```
src/models.py:5: class User:
src/models.py:15: class Product:
src/utils.py:10: class Helper:
```

**Agent reasoning:** "I found classes in `models.py` and `utils.py`. Let me check their sizes before loading."

**Context used:** ~100 lines

---

### Step 3: Check File Metadata

```python
# Check file sizes to avoid loading huge files
for file in ['src/models.py', 'src/utils.py']:
    info = get_file_info(file)
    print(f"{file}: {info['size_bytes']} bytes")
```

**Output (example):**
```
src/models.py: 2450 bytes (OK to load)
src/utils.py: 1800 bytes (OK to load)
```

**Agent reasoning:** "Both files are small enough to load fully. If they were very large, I'd use `extract_function_signatures` instead."

**Context used:** ~10 lines

---

### Step 4: Extract Function Signatures (Optional for Large Files)

```python
# For large files, extract signatures first to understand structure
signatures = extract_function_signatures('src/models.py', 'python')
print(signatures)
```

**Output (example):**
```
class User:
    def __init__(self, name, email):
class Product:
    def __init__(self, title, price):
    def calculate_discount(self):
```

**Agent reasoning:** "I can see the `__init__` methods and their parameters without loading the full implementations. Now I know exactly what I need to modify."

**Context used:** ~50 lines

---

### Step 5: Load Only Necessary Files

```python
# Now load just the relevant files
models_content = load_string_from_file('src/models.py')
utils_content = load_string_from_file('src/utils.py')
```

**Context used:** ~150 lines (the actual file content)

---

### Step 6: Make Targeted Changes

```python
# Make specific changes using replace_on_file
replace_on_file(
    'src/models.py',
    'def __init__(self, name, email):',
    'def __init__(self, name: str, email: str):'
)

replace_on_file(
    'src/models.py',
    'def __init__(self, title, price):',
    'def __init__(self, title: str, price: float):'
)
```

**Context used:** ~20 lines

---

### Step 7: Verify Changes

```python
# Compare original with modified
diff = compare_files('src/models.py.bak', 'src/models.py')
print(diff)
```

**Output (example):**
```diff
--- src/models.py.bak
+++ src/models.py
@@ -5,7 +5,7 @@
 class User:
-    def __init__(self, name, email):
+    def __init__(self, name: str, email: str):
         self.name = name
         self.email = email

@@ -15,7 +15,7 @@
 class Product:
-    def __init__(self, title, price):
+    def __init__(self, title: str, price: float):
         self.title = title
         self.price = price
```

**Agent reasoning:** "Perfect! The changes look correct."

**Context used:** ~30 lines

---

## Total Context Comparison

### Traditional Approach: 100,000+ lines ❌
### New Approach: ~410 lines ✅

**Savings: 99.6% less context used!** 🎉

---

## Real Agent Conversation Example

Here's how an actual conversation with a context-efficient agent might look:

**User:** "Find all Python files that define classes and add type hints to their `__init__` methods."

**Agent:**
```
<thoughts>
I should use context-efficient tools to solve this without loading all files.
First, I'll survey the project structure, then search for class definitions,
and finally load and modify only the relevant files.
</thoughts>

<runcode>
# Step 1: Survey the project
tree = list_directory_tree('.', max_depth=2, show_files=False)
print("Project structure:")
print(tree)
</runcode>
```

**Execution logs:**
```
Project structure:
project/
├── src/
├── tests/
└── docs/
```

**Agent:**
```
<thoughts>
Good, the main code is in src/. Let me search for class definitions there.
</thoughts>

<runcode>
# Step 2: Find files with class definitions
results = search_in_files('src', 'class ', file_extensions=('.py',))
print("Found classes:")
print(results)
</runcode>
```

**Execution logs:**
```
Found classes:
src/models.py:5: class User:
src/models.py:15: class Product:
src/utils.py:10: class Helper:
```

**Agent:**
```
<thoughts>
Found classes in models.py and utils.py. Let me check their sizes
and extract signatures to understand what needs to be modified.
</thoughts>

<runcode>
# Step 3: Check file sizes
for file in ['src/models.py', 'src/utils.py']:
    info = get_file_info(file)
    print(f"{file}: {info['size_bytes']} bytes")

# Step 4: Extract signatures
print("\nSignatures in models.py:")
sigs = extract_function_signatures('src/models.py', 'python')
print(sigs)
</runcode>
```

**Execution logs:**
```
src/models.py: 2450 bytes
src/utils.py: 1800 bytes

Signatures in models.py:
class User:
    def __init__(self, name, email):
class Product:
    def __init__(self, title, price):
```

**Agent:**
```
<thoughts>
Perfect! Both files are small. I can see the __init__ methods need type hints.
Let me load and modify them.
</thoughts>

<runcode>
# Step 5: Load and modify
models = load_string_from_file('src/models.py')

# Add type hints to User.__init__
replace_on_file(
    'src/models.py',
    'def __init__(self, name, email):',
    'def __init__(self, name: str, email: str):'
)

# Add type hints to Product.__init__
replace_on_file(
    'src/models.py', 
    'def __init__(self, title, price):',
    'def __init__(self, title: str, price: float):'
)

# Do the same for utils.py
utils_sigs = extract_function_signatures('src/utils.py', 'python')
print("Modified files successfully!")

final_answer("I've added type hints to the __init__ methods in src/models.py and src/utils.py. All class constructors now have proper type annotations.")
</runcode>
```

---

## Key Benefits Demonstrated

1. **Context Efficiency** 
   - Used 410 lines vs 100,000+ lines
   - 99.6% context savings

2. **Speed**
   - No need to load and parse massive amounts of code
   - Faster execution, lower costs

3. **Intelligence**
   - Agent understands project structure before diving in
   - Makes informed decisions about what to load
   - Strategic, not brute force

4. **Scalability**
   - Can work with much larger codebases
   - Stays within context limits
   - Handles projects that would otherwise be impossible

---

## Conclusion

The context-efficient tools transform how AI agents work with code:

- **Before:** Load everything, hope for the best ❌
- **After:** Survey, search, load strategically ✅

This is the difference between an agent that can handle a 1,000-line project and one that can handle a 100,000-line project.
