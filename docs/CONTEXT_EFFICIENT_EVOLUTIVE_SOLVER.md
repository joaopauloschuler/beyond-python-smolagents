# Context-Efficient evolutive_problem_solver_folder

## Overview

The `evolutive_problem_solver_folder` function in `src/smolagents/bp_thinkers.py` has been optimized to be context-efficient. This modification addresses the issue of excessive context consumption when evaluating multiple solution folders during the evolutionary problem-solving process.

## The Problem

Previously, the function loaded entire solution folders using `source_code_to_string()` at each evolutionary step:

```python
# OLD APPROACH - Context inefficient
task_description = """
<solution1>"""+source_code_to_string('solution1/')+"""</solution1>
<solution2>"""+source_code_to_string('solution2/')+"""</solution2>
<solution3>"""+source_code_to_string('solution3/')+"""</solution3>
"""
```

**Issues:**
- For a 50,000-line codebase across 3 solutions, this would load ~150,000 lines into context
- Consumed massive amounts of context window
- Hit context limits on larger projects
- Wasteful since most content wasn't relevant for high-level comparison

## The Solution

The function now provides structure and metrics instead of full content:

```python
# NEW APPROACH - Context efficient
solution1_tree = list_directory_tree('solution1/', max_depth=3, show_files=True)
solution1_metrics = count_lines_of_code('solution1/')

task_description = """
<solution1_structure>
"""+solution1_tree+"""
Lines of code metrics: """+str(solution1_metrics)+"""
</solution1_structure>
"""
```

**Benefits:**
- Provides directory structure (shows file organization)
- Provides code metrics (shows scale and composition)
- Agent can use context-efficient tools to inspect specific parts on-demand
- Maintains all functionality while drastically reducing context usage

## Context Savings

Based on integration tests with sample solutions:
- **Old approach**: 5,352 characters for 3 solutions
- **New approach**: 399 characters for 3 solutions
- **Savings**: 92.5% (13.4x reduction)

For real-world projects, savings scale proportionally to project size.

## Tools Available to Agents

The agent receives guidance on using context-efficient tools:

1. **`list_directory_tree(folder, max_depth, show_files)`** - See folder structure without loading content
2. **`search_in_files(folder, pattern, file_extensions)`** - Find specific code patterns efficiently
3. **`extract_function_signatures(filename, language)`** - Understand APIs without full implementations
4. **`get_file_info(filepath)`** - Check file sizes before loading
5. **`count_lines_of_code(folder)`** - Understand project scale
6. **`load_string_from_file(filename)`** - Load full content only when truly needed

## Example Usage

The function works the same way as before:

```python
from smolagents import LiteLLMModel
from smolagents.bp_thinkers import evolutive_problem_solver_folder, DEFAULT_THINKER_TOOLS

model = LiteLLMModel(model_id="gemini/gemini-2.5-flash", api_key="your-key")

evolutive_problem_solver_folder(
    model,
    task_str="Create a task management application",
    agent_steps=54,
    steps=4,
    start_now=True,
    tools=DEFAULT_THINKER_TOOLS
)
```

The agent now:
1. Receives structure and metrics for each solution
2. Can inspect specific files using context-efficient tools
3. Makes informed decisions about which solution is better
4. Only loads full file content when necessary

## Technical Details

**Modified sections:**
- Lines 668-706 in `src/smolagents/bp_thinkers.py`

**Key changes:**
1. Added computation of directory trees before creating task description
2. Added computation of code metrics before creating task description
3. Replaced full content embedding with structure/metrics embedding
4. Added guidance to agent on using context-efficient tools

**Backward compatibility:**
- Function signature unchanged
- Return value unchanged
- All functionality preserved
- Existing code using this function continues to work

## Testing

Integration tests verify:
1. ✅ Context savings of >80% (achieved 92.5%)
2. ✅ Useful information still provided (structure + metrics)
3. ✅ No regression in existing tests

Run tests:
```bash
# Run context-efficiency tests
python -m pytest tests/test_evolutive_solver_context.py -v

# Run all context tools tests
python -m pytest tests/test_bp_context_tools.py -v
```

## Best Practices

When using `evolutive_problem_solver_folder`:

1. **Use DEFAULT_THINKER_TOOLS** - Includes all context-efficient tools
2. **Keep `max_depth=3`** - Good balance between overview and detail
3. **Trust the agent** - It can inspect specific files when needed
4. **Monitor context usage** - Check if you're approaching limits

## Future Enhancements

Potential improvements:
- Make `max_depth` configurable as a parameter
- Add option to include file signatures in initial summary
- Add caching for frequently accessed files
- Provide diff-based updates instead of full content in iterations

## Related Documentation

- [Context-Efficient Tools Guide](../../bp-examples/CONTEXT_EFFICIENT_TOOLS.md)
- [Context-Efficient Tools Demo](../../bp-examples/agent-using-context-efficient-tools.md)
- [Main README](../../README.md)
