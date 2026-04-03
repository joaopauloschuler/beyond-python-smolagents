<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# BPSA - Beyond Python SmolAgents

![How to Install BPSA](docs/img/BPSA-HowToInstall.gif?raw=true)


**BPSA - Beyond Python SmolAgents** is a fork of the original [smolagents](https://github.com/huggingface/smolagents) that extends its original abilities:
* 💻 **Interactive CLI ([`bpsa`](#cli-bpsa)):** Multi-turn REPL with slash commands, command history, tab completion, session stats, and auto-approve mode.
* 🔄 **Infinite runtime CLI ([`ad-infinitum`](#cli-ad-infinitum)):** Allows agents to **run ad infinitum** via autonomous looping.
* 🗜️ **Context compression**: Biologically inspired [automatic LLM-based summarization](docs/compression.md) of older memory steps to manage context window size during long-running tasks.
* 🌐 **Browser integration:** Control a headed Chromium browser from agent code blocks via Playwright (`--browser` flag).
* 🖥️ **GUI interaction:** Launch, screenshot, click, type, and send keys to native GUI applications on X11 via xdotool/ImageMagick (`--gui-x11` flag).
* 🔌 **MCP server integration:** Connect any [Model Context Protocol](https://modelcontextprotocol.io) server as a tool source via the `--mcp` CLI flag. Supports both HTTP (Streamable HTTP) and stdio-based servers.
* 👁️ **Image loading:** Agents can load and visually inspect image files (plots, screenshots, diagrams) via the built-in `load_image` tool — always available, no flags needed.
* 🎨 **Image tools:** Visual image diffing (`diff_images`), OCR text extraction from images (`screen_ocr`), and a canvas for drawing shapes, text, and annotations (`canvas_create`, `canvas_draw`) — always available.
* 🎤 **Dictation input:** Dictate prompts via microphone using Whisper or ElevenLabs transcription (`/dictation` command, requires `BPSA_DICTATION_TRANSCRIBER` env var).
* ⚡ **Native Python execution:** Execute Python code natively via `exec` for unrestricted processing.
* 🌍 **Multi-language support:** Code in multiple languages beyond Python (Pascal, PHP, C++, Java and more).
* 🛠️ **Developer tools:** Lots of new tools that help agents to compile, test, and debug source code in various computing languages.
* 👥 **Multi-agent collaboration:** Collaborate across multiple agents to solve complex problems.
* 🔍 **Research tools:** Tools that help agents to research and write technical documentation.
* 📚 **Documentation generation:** Generate and update documentation including READMEs for existing codebases.


## Installation
Install the project, including the dictation support, CLIs, OpenAI protocol and LiteLLM dependencies.

```bash
$ pip install bpsa[dictation,browser,openai,litellm]
```

This will set up the necessary libraries and the Beyond Python Smolagents framework in your environment.

## CLI (`bpsa`)

Beyond Python Smolagents includes an interactive CLI called `bpsa`. It provides a multi-turn REPL powered by `CodeAgent` with all `DEFAULT_THINKER_TOOLS` and context compression enabled.

### Environment Variables

Configure `bpsa` via environment variables or a `.env` file in your working directory:

Supported model classes: `OpenAIServerModel`, `LiteLLMModel`, `LiteLLMRouterModel`, `InferenceClientModel`, `TransformersModel`, `AzureOpenAIServerModel`, `AmazonBedrockModel`, `VLLMModel`, `MLXModel`, `GoogleColabModel`.

Example `.env` file:
```
BPSA_SERVER_MODEL=OpenAIServerModel
BPSA_API_ENDPOINT=https://api.poe.com/v1
BPSA_KEY_VALUE=your_api_key
BPSA_MODEL_ID=Gemini-2.5-Flash
BPSA_MAX_TOKENS=64000
```

Context compression parameters can also be configured via env vars (e.g., `BPSA_COMPRESSION_ENABLED`, `BPSA_COMPRESSION_KEEP_RECENT_STEPS`). See [CLI.md](docs/CLI.md) for the full list.

#### Dictation Input

Dictate prompts via microphone instead of typing. Requires the dictation extra and a transcriber environment variable:

```bash
pip install bpsa[dictation]

# Option 1: Whisper (local, offline)
export BPSA_DICTATION_TRANSCRIBER=whisper
export BPSA_DICTATION_MODEL=base.en        # optional (default: base.en)

# Option 2: ElevenLabs (cloud API)
export BPSA_DICTATION_TRANSCRIBER=elevenlabs
export ELEVENLABS_API_KEY=your_api_key
```

Then use `/dictation on` in the REPL to start listening and `/dictation off` to stop. While active, the prompt shows `[mic] >` and transcribed speech is inserted at the cursor.

### BPSA CLI Usage

```bash
$ bpsa                              # Interactive REPL (default)
$ bpsa run "task description"       # One-shot mode
$ echo "task" | bpsa                # Piped input
$ bpsa --load-instructions          # Load CLAUDE.md, AGENTS.md, etc. at startup
$ bpsa --browser                    # Enable Playwright browser integration
$ bpsa --gui-x11                     # Enable native GUI interaction (xdotool/ImageMagick)
$ bpsa --image                       # Enable image analysis and drawing tools
$ bpsa --mcp http://localhost:8000/mcp  # Connect an HTTP MCP server
$ bpsa --mcp 'npx -y @modelcontextprotocol/server-filesystem /'  # Connect a stdio MCP server
```

The REPL supports command history, tab completion for slash commands, and multi-line input via Alt+Enter. Use `/session-save <file>` and `/session-load <file>` to persist and restore sessions across restarts. You can also launch `ad-infinitum` from within the REPL via `!ad-infinitum ...`. Type `/help` to see all available commands.

#### Shell commands from the REPL

| Prefix | Description |
|--------|-------------|
| `!<command>` | Run an OS command directly (agent does not see the output) |
| `!!<command>` | Run an OS command with streaming output; output is appended to the next prompt sent to the agent |
| `!!!<command>` | Run an OS command and immediately send the output to the agent for analysis |

#### Aliases

Define command aliases with `/alias <name> <value>` (e.g., `/alias gs !!git status`). Aliases are saved to `~/.bpsa_aliases` and persist across sessions. Use `/alias` to list all and `/alias -d <name>` to delete.

#### Auto-save

Sessions are automatically saved every 5 turns to `~/.bpsa_autosave.json`. Configure the interval with the `BPSA_AUTOSAVE_INTERVAL` environment variable (set to 0 to disable).

Find more about bpsa CLI at [CLI.md](docs/CLI.md).

## CLI (`ad-infinitum`)

`ad-infinitum` is a dedicated CLI for autonomous, looping agent execution. It loads tasks from a folder of task files (`.md`, `.py`, `.sh`) or a single file and runs them repeatedly.

- **`.md` files** are treated as agent prompts (run via `agent.run()`)
- **`.py` files** are executed directly via the Python interpreter (`subprocess`)
- **`.sh` files** are executed directly via bash (`subprocess`)

Script files (`.py`, `.sh`) bypass the agent entirely, enabling mixed workflows where setup, validation, and cleanup steps run as plain scripts alongside agent-driven prompt tasks.

### How It Works

Each cycle iterates through all tasks in order.

### Task Folder Convention

```
tasks/
+-- _preamble.md          (optional) prepended to ALL prompt tasks
+-- 01-setup-env.sh       script: install deps, create dirs
+-- 02-implement.md       prompt: agent does the work
+-- 03-validate.py        script: programmatic validation
+-- 04-refine.md          prompt: agent fixes issues
+-- _postamble.md         (optional) appended to ALL prompt tasks
```

- Files starting with `_` are **modifiers**, not tasks
- `_preamble.md` is prepended to every **prompt** task (e.g., project context, coding standards)
- `_postamble.md` is appended to every **prompt** task (e.g., "commit when done", "call final_answer with a summary")
- All other `.md`, `.py`, and `.sh` files are tasks, loaded in **alphabetical order**
- Numbering prefixes (`01-`, `02-`) give natural sequencing
- Script tasks (`.py`, `.sh`) are executed directly and report exit codes instead of token usage

### Usage

```bash
$ ad-infinitum ../tasks/              # Run all task files from a folder
$ ad-infinitum ../single-task.md      # Run a single prompt task
$ ad-infinitum ../setup.sh            # Run a single shell script
$ ad-infinitum ../validate.py         # Run a single Python script
$ ad-infinitum ../tasks/ -c 5         # Run 5 cycles
$ ad-infinitum ../tasks/ --cycles 0   # Run ad infinitum
```

| Flag | Description |
|---|---|
| `-c`, `--cycles` | Number of cycles, 0 = infinite (overrides `BPSA_CYCLES`) |

### Environment Variables

`ad-infinitum` uses the same `BPSA_*` environment variables as `bpsa`, plus these additional ones:

| Variable | Default | Description |
|---|---|---|
| `BPSA_CYCLES` | `1` | Number of cycles (0 = infinite) |
| `BPSA_MAX_STEPS` | `200` | Max steps per agent run |
| `BPSA_PLAN_INTERVAL` | off | Planning interval (e.g., `22`) |
| `BPSA_COOLDOWN` | `0` | Seconds to wait between cycles |
| `BPSA_INJECT_FOLDER` | `true` | Inject directory tree (see `bpsa` section above). Only applies to `.md` prompt tasks. |

Example `.env` file:
```
BPSA_SERVER_MODEL=OpenAIServerModel
BPSA_API_ENDPOINT=https://api.poe.com/v1
BPSA_KEY_VALUE=your_api_key
BPSA_MODEL_ID=Gemini-2.5-Flash
BPSA_CYCLES=3
BPSA_INJECT_FOLDER=true
BPSA_MAX_STEPS=200
BPSA_COOLDOWN=5
```

### Execution Model

With 4 task files and `BPSA_CYCLES=2`:

```
Cycle 1/2:
  Task 1/4: 01-setup-env.sh     (script, runs via bash)
  Task 2/4: 02-implement.md     (prompt, fresh agent)
  Task 3/4: 03-validate.py      (script, runs via python)
  Task 4/4: 04-refine.md        (prompt, fresh agent, sees files from earlier tasks)
Cycle 2/2:
  Task 1/4: 01-setup-env.sh     (script, re-runs setup)
  Task 2/4: 02-implement.md     (prompt, fresh agent, sees evolved project)
  Task 3/4: 03-validate.py      (script, re-validates)
  Task 4/4: 04-refine.md        (prompt, fresh agent)
```

### Graceful Shutdown

- **Single Ctrl+C**: Finishes the current task, then stops
- **Double Ctrl+C**: Aborts immediately


## The Thinkers
There are 2 main functions that you can easily call:
* [fast_solver](https://github.com/joaopauloschuler/beyond-python-smolagents?tab=readme-ov-file#the-fast_solver) : A multi-agent parallel problem-solving approach that generates 3 independent solutions using different AI models, then synthesizes them into an optimized final solution. Think of it as automated "brainstorming → best-of-breed synthesis" that leverages diverse AI perspectives for higher quality outcomes.

  [![Watch the video](docs/img/writing-process.jpg?raw=true)](https://youtu.be/oQ2GdrtWR94)

* [evolutive_problem_solver](https://github.com/joaopauloschuler/beyond-python-smolagents?tab=readme-ov-file#the-heavy-thinker---evolutive_problem_solver) : An iterative evolutionary approach that refines solutions through multiple generations, using analysis, comparison, mixing, and improvement cycles with accumulated knowledge. It mimics natural selection where solutions compete, combine, and evolve over time to converge on increasingly better results.

  [![Watch the video](docs/img/evol.jpg?raw=true)](https://youtu.be/XuFL3PQGQkc)

  [![Watch the video](docs/img/monologue.jpg?raw=true)](https://youtu.be/25uJ0VHDKZE)

## Google colab ready to run examples 

### Writing task examples
* [Write about the importance of vitamin C - `fast_solver`](https://colab.research.google.com/github/joaopauloschuler/beyond-python-smolagents/blob/v1.24-bp/bp-examples/writing/vitamin-C-with-fast-solver.ipynb)
* [Write about the importance of vitamin C - `fast_solver using 3 models working together`](https://colab.research.google.com/github/joaopauloschuler/beyond-python-smolagents/blob/v1.24-bp/bp-examples/writing/vitamin-C-with-fast-solver-3-models-work-together.ipynb)
* [Write about the importance of vitamin C - `evolutive_problem_solver`](https://colab.research.google.com/github/joaopauloschuler/beyond-python-smolagents/blob/v1.24-bp/bp-examples/writing/vitamin-C.ipynb)

### Coding task examples
  [![Watch the video](docs/img/coding-example.jpg?raw=true)](https://youtu.be/0EronXSvJDs)
* [In C++, code a task manager - `evolutive_problem_solver`](https://colab.research.google.com/github/joaopauloschuler/beyond-python-smolagents/blob/v1.24-bp/bp-examples/cpp/cpp-single-file-01.ipynb)
* [In PHP, code a task manager - `evolutive_problem_solver`](https://colab.research.google.com/github/joaopauloschuler/beyond-python-smolagents/blob/v1.24-bp/bp-examples/php/php-single-file-01.ipynb)
* [In Java, code a task manager - `evolutive_problem_solver`](https://colab.research.google.com/github/joaopauloschuler/beyond-python-smolagents/blob/v1.24-bp/bp-examples/java/java-single-file-01.ipynb)
* [In Free Pascal, code a task manager - `evolutive_problem_solver`](https://colab.research.google.com/github/joaopauloschuler/beyond-python-smolagents/blob/v1.24-bp/bp-examples/pascal/pascal-single-file-01.ipynb) 
* [Create a readme - `evolutive_problem_solver`](https://colab.research.google.com/github/joaopauloschuler/beyond-python-smolagents/blob/v1.24-bp/bp-examples/writing/source_code_documentation_pascal.ipynb)

## Basic usage (single agent)
Create a single agent with various tools for working with different programming languages:
```
import smolagents
from smolagents.bp_tools import *
from smolagents.bp_utils import *
from smolagents.bp_thinkers import *
from smolagents import LiteLLMModel, LogLevel
from smolagents import CodeAgent, MultiStepAgent, ToolCallingAgent
from smolagents import tool

MAX_TOKENS = 64000
coder_model_id = "gemini/gemini-2.5-flash"
coder_model = LiteLLMModel(model_id=coder_model_id, api_key=YOUR_KEY_VALUE, max_tokens=MAX_TOKENS)

tools = [ run_os_command,
  copy_file, is_file, 
  print_file_lines, get_line_from_file, count_file_lines,
  read_file_range, insert_lines_into_file, replace_line_in_file,
  remove_pascal_comments_from_string, pascal_interface_to_string,
  source_code_to_string, string_to_source_code,
  run_os_command, replace_in_file, replace_in_file_from_files,
  get_file_size, load_string_from_file, save_string_to_file, append_string_to_file,
  list_directory_tree, search_in_files, get_file_info, list_directory,
  extract_function_signatures, compare_files, count_lines_of_code,
  mkdir, delete_file, delete_directory, compare_folders
  ]

coder_agent = CodeAgent( model=coder_model, tools = tools, add_base_tools=True)
coder_agent.run("Please list the files in the current folder.")
```

## Context Compression

For long-running tasks with many steps, agent memory can grow large and exceed context window limits. Context compression automatically summarizes older memory steps via LLM while keeping recent steps in full detail.

### Basic Usage

```python
from smolagents import CodeAgent, CompressionConfig, LiteLLMModel

model = LiteLLMModel(model_id="gemini/gemini-2.5-flash", api_key=YOUR_KEY)

# Configure compression
config = CompressionConfig(
    keep_recent_steps=5,       # Keep last 5 steps in full detail
    max_uncompressed_steps=10,   # Compress when step count exceeds 10
)

# Create agent with compression enabled
agent = CodeAgent(
    model=model,
    tools=tools,
    compression_config=config,
)

agent.run("Complex multi-step task...")
```

### Using a Cheaper Model for Compression

To reduce costs, you can use a smaller/cheaper model for the compression summarization:

```python
main_model = LiteLLMModel(model_id="gemini/gemini-2.5-pro", api_key=YOUR_KEY)
compression_model = LiteLLMModel(model_id="gemini/gemini-2.5-flash", api_key=YOUR_KEY)

config = CompressionConfig(
    keep_recent_steps=5,
    max_uncompressed_steps=8,
    compression_model=compression_model,  # Use cheaper model for compression
)

agent = CodeAgent(
    model=main_model,
    tools=tools,
    compression_config=config,
)
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `True` | Enable/disable compression |
| `keep_recent_steps` | `5` | Number of recent steps to keep in full detail |
| `max_uncompressed_steps` | `10` | Trigger compression when step count exceeds this |
| `max_compressed_steps` | `32` | Merge compressed summaries when count exceeds this (0 = disabled) |
| `keep_compressed_steps` | `22` | Number of recent compressed summaries to keep during merge |
| `estimated_token_threshold` | `0` | Trigger based on estimated tokens (0 = disabled) |
| `compression_model` | `None` | Optional separate model for compression |
| `preserve_error_steps` | `False` | Always keep steps with errors |
| `preserve_final_answer_steps` | `True` | Always keep final answer steps |
| `min_compression_chars` | `4096` | Minimum chars before compression LLM call is made (0 = disabled) |

When using `bpsa` or `ad-infinitum`, all of the above can be configured via environment variables (e.g., `BPSA_COMPRESSION_ENABLED`, `BPSA_COMPRESSION_KEEP_RECENT_STEPS`) without changing any code. See [CLI.md](docs/CLI.md) for the full list.


### What Gets Preserved

The compression system always preserves:
- The original task (TaskStep)
- Recent N steps (configured via `keep_recent_steps`)
- Steps with errors (helps agent learn from mistakes)
- Final answer steps

Older action and planning steps are summarized into a `CompressedHistoryStep` that captures key decisions, observations, and progress. When compressed summaries accumulate beyond `max_compressed_steps`, the older ones are merged while `keep_compressed_steps` most recent summaries are preserved at full fidelity.

## The `fast_solver`
The `fast_solver` function is a sophisticated multi-agent problem-solving approach that leverages the "wisdom of crowds" principle with AI models.

### Core Purpose
This function takes a complex task and solves it by generating multiple independent solutions, then intelligently combining them into a superior final solution.

### Workflow Breakdown

#### Phase 1: Independent Solution Generation
1. **Creates 3 separate AI agents** using potentially different models (`p_coder_model`, `p_coder_model2`, `p_coder_model3`)
2. **Each agent independently solves the same task** without knowledge of the others' work
3. **Saves each solution to separate files** (`solution1.ext`, `solution2.ext`, `solution3.ext`)
4. **Includes fallback logic** - if an agent fails to save its solution initially, it gets a second chance

#### Phase 2: Solution Synthesis
1. **Loads all three solutions** from the saved files
2. **Creates a fourth "final" agent** (using `p_coder_model_final`)
3. **Presents all three solutions to this agent** with instructions to mix and combine the best parts
4. **Generates a final optimized solution** that synthesizes the strengths of all previous attempts

### Key Features

**Multi-Model Support**: Can use up to 4 different AI models - allowing you to leverage different models' strengths (e.g., one model might be better at creativity, another at technical accuracy).

**Robust Error Handling**: If any agent fails to save its solution initially, the function automatically retries.

**Flexible Output**: The `fileext` parameter allows generating different types of content (code files, documentation, etc.).

**Rich Motivation**: Each agent receives encouraging prompts to "show your intelligence with no restraints" and produce extensive, detailed solutions.

### Why This Approach Works

1. **Diversity**: Multiple independent attempts often explore different solution approaches
2. **Quality Enhancement**: The final synthesis stage can identify and combine the best elements from each approach
3. **Error Mitigation**: If one agent produces a poor solution, the others can compensate
4. **Scalability**: Can leverage different specialized models for different aspects of the problem

This is essentially an automated "brainstorming → synthesis" workflow that mimics how human teams might approach complex problems.

## The heavy thinker - `evolutive_problem_solver`
Using "Heavy Thinking" is typically more computationally intensive and time-consuming than basic single-agent tasks, but it is designed to yield superior results for difficult problems that benefit from a more thorough, multi-pass approach.
`evolutive_problem_solver` combines evolutive computing, genetic algorithms and agents to produce a final result.

The "Heavy Thinking" method within Beyond Python Smolagents represents an advanced paradigm for tackling highly complex or open-ended problems that may not be solvable in a single agent turn. It's particularly useful for tasks requiring significant iterative refinement, exploration, or multi-step reasoning, such as generating comprehensive documentation from a large codebase or complex coding tasks.

While `evolutive_problem_solver` internal workings involve sophisticated logic, the user interacts with it by providing a detailed task prompt and a set of tools. `evolutive_problem_solver` has an iterative process, potentially involving multiple agent interactions, intermediate evaluations, and refinements over several "steps" and "agent_steps" within each step, aiming to converge on a high-quality solution.

Here is how you might conceptually set up and invoke the `evolutive_problem_solver` for a task like generating comprehensive documentation from source code. This example focuses on *how* you would structure the input prompt and call the function:

```
!git clone git@github.com:joaopauloschuler/neural-api.git
current_source = source_code_to_string('neural-api')
project_name = 'neural-api'
task = """You have access to an Ubuntu system. You have available to you python, php and free pascal.
You are given the source code of the """+project_name+""" project in the tags <file filename="..."> source code file content </file>.
This is the source code:"""+current_source+"""
Your highly important and interesting task is producing a better version of the README.md file.
You will save the updated versions of the README.md into new files as directed.
The original version of the readme file is provided in the tag <file filename="README.md"></file>.
When asked to test, given that this is a task regarding documentation, you should review the README file.
When asked to code, you will produce documentation.

You will write the documentation in a technical and non commercial language.
You contribution will be helping others to understand how to use this project and its inner workings so future
developers will be able to build on the top of it.
It would be fantastic if you could add to the documentation ideas about to solve real world problems using this project.
For saving documentation, use the tags <savetofile> and <appendtofile>. Trying to save documentation via python code is just too hard and error prone.
When asked to test or review documentation, make sure that referred files or functions do actually exist. This is to prevent broken links.
Your documentation should focus on existing features only. Do not document future or to be be developed features.
Your goal is documentation.
Avoid adding code snippets.
"""
print("Input size:", len(task))
# Run the evolutive solver
evolutive_problem_solver(
    coder_model,       # The LLM to use
    task,              # The task description
    agent_steps=54,    # Number of steps each agent can take
    steps=4,           # Number of evolutionary iterations
    start_now=True,    # Start from scratch
    fileext='.md',     # File extension for outputs
    tools=tools        # Tools available to the agents
)
```

The source code above shows one of the core strengths of Beyond Python Smolagents: Its ability to work with codebases across multiple languages to generate and update documentation automatically. The `source_code_to_string` and `pascal_interface_to_string` tools are particularly useful here, allowing agents to ingest the codebase structure and content.

For complex documentation tasks, such as generating a comprehensive README from a large project, you should leverage advanced techniques provided by `evolutive_problem_solver`.

### Heavy thinking inner workings

**1. Overall Workflow:**

The `evolutive_problem_solver` function sets up a loop where a `CodeAgent` acts as both a coder and a critic. It starts with initial solutions, then enters a cycle of:
1.  Analyzing and comparing current solutions.
2.  Potentially mixing solutions if beneficial.
3.  Selecting the "best" current solution.
4.  Generating two new alternative solutions by applying improvements suggested by the agent itself, potentially guided by past advice.
5.  Refining the new solutions (detailing changes, testing, getting advice).
6.  Potentially merging smaller new solutions with the current best.

This process simulates an evolutionary cycle where solutions compete, combine (mixing), and are refined based on criteria evaluated by the AI agent, aiming to improve the quality of the solution over time. The `advices.notes` file serves as a form of accumulated knowledge or 'genetic memory' for the agent across iterations. The process repeats for a fixed number of `steps`.

**2. `get_local_agent()` Inner Function:**

This helper function is responsible for creating and configuring a `CodeAgent` instance based on the parameters passed to the main `evolutive_problem_solver` function. It sets up the agent's tools, model, import permissions, max steps, callbacks, executor type, system prompt, and log level. This ensures that a fresh agent instance with the desired configuration is available whenever needed during the process.

**3. `test_and_refine(local_agent, solution_file)` Inner Function:**

This function orchestrates a series of refinement steps for a given `solution_file` using the `local_agent`. It guides the agent through the following tasks:
*   **Refine 1:** Prompts the agent to detail the changes it made (presumably in the immediately preceding step where the solution file was created or modified).
*   **Refine 2:** Instructs the agent to review and test its own solution. If the agent feels it needs further refinement, it's prompted to update the full source code in the specified `solution_file` and call `final_answer("Task completed! YAY!")`.
*   **Refine 3:** Asks the agent to provide any advice it would give to its future self based on the current task and solution process. The output of this step is captured as `new_advice`. If `new_advice` is not empty, it is appended to a file named `advices.notes`, separated by a horizontal rule (`---`).

**4. Main Execution Logic:**

*   **Initialization:**
    *   A `local_task_description` is created, wrapping the original `task_str` in `<task>` tags.
    *   A list `valid_solutions` is defined to hold the base filenames for the three potential solutions ('solution1', 'solution2', 'solution3').
    *   A `motivation` string is defined, encouraging the agent to be extensive, detailed, and creative.

*   **Initial Solution Generation (`if start_now:`):**
    *   If `start_now` is True, the process begins by generating the first three distinct solutions.
    *   A `local_agent` is obtained using `get_local_agent()`.
    *   The agent is run three times, each time tasked with solving the `local_task_description` with the added `motivation` and saving the output to `solution1`, `solution2`, and `solution3` respectively (with the specified `fileext`). The `reset=True` ensures each initial generation starts with a fresh context for the agent.
    *   After each initial solution is generated, `test_and_refine` is called for that solution file to detail changes, test, and capture advice.

*   **Evolution Loop (`for i in range(steps):`):**
    *   The code enters a loop that runs for `steps` iterations, representing the evolutionary process.
    *   Inside the loop, a new `local_agent` is created at the start of each iteration.
    *   **Analysis and Comparison:**
        *   A detailed `task_description` is created. This prompt includes the original task, the content of `solution1`, `solution2`, and `solution3` (loaded using `load_string_from_file`), all enclosed in appropriate XML-like tags (`<solution1>`, etc.).
        *   The agent is instructed to analyze these three solutions, explain their advantages and disadvantages, prioritize solutions with more features, and output the analysis as text using `final_answer()`. The agent is explicitly told *not* to code anything except calling `final_answer` with text.
        *   The agent is run with this analysis task (`reset=True` for a fresh start).
    *   **Mixing Decision:**
        *   If it's not one of the last two steps (`i < steps - 2`), the agent is asked if mixing parts of the solutions would be beneficial.
        *   The agent's response is captured, and if it's 'yes', the `should_mix` flag is set to True.
    *   **Mixing Logic (`if should_mix:`):**
        *   If mixing is deemed beneficial, `solution2` is chosen as the destination file for the mixed solution.
        *   The agent is tasked with mixing parts of the existing solutions into `solution2` and saving the full result there.
        *   `test_and_refine` is called on `solution2`.
        *   `continue` skips the rest of the current loop iteration (selection and alternative generation) and proceeds to the next evolutionary step with the potentially mixed `solution2` now available for comparison.
    *   **Best Solution Selection:**
        *   If mixing is not happening, the agent is asked to select the best solution among `solution1`, `solution2`, and `solution3` by calling `final_answer()` with the chosen filename.
        *   The selected filename is stored in `selected_solution`.
    *   **Best Solution Handling:**
        *   If `selected_solution` is one of the valid filenames, the corresponding file is copied to `best_solution.best`.
        *   **Alternative Generation and Improvement (`if i < steps - 1:`):**
            *   If it's not the very last step (`i < steps - 1`), the process prepares to generate alternative solutions based on the newly selected `best_solution.best`.
            *   The current `best_solution.best` is copied to `solution3` to serve as a baseline for comparison in the next iteration.
            *   A loop runs twice (for `alternatives_cnt` 0 and 1), targeting `solution1` and `solution2` as the files for the new alternatives.
            *   For each alternative:
                *   The agent is presented with the current `best_solution.best` and the accumulated `advices.notes` and asked to suggest improvements (outputting text via `final_answer`).
                *   The agent is asked to randomly pick one suggestion and code it.
                *   The agent is run to implement the selected improvement, tasked with updating the *existing* solution rather than starting a new one. For the first alternative (`alternatives_cnt == 0`), the agent is encouraged to be bold and add many improvements.
                *   The agent is asked if more review/testing is needed.
                *   The agent is instructed to save the *full* updated solution to the current `solution_file` (`solution1` or `solution2`) using `<savetofile>` tags and confirm completion with `final_answer("Task completed! YAY!")`.
                *   `test_and_refine` is called on this updated solution file.
                *   **Merging Smaller Solutions:** A peculiar step checks if the newly generated `solution_file` is *smaller* than the `best_solution.best`. If it is, the agent is tasked with merging the `best_solution.best` and the new `solution_file`, assuming the larger `best_solution.best` might contain valuable parts missing from the smaller new version. The merged result is saved back to the `solution_file`.
    *   **Error Handling:** A `try...except` block is present to catch potential errors during the loop iteration, printing 'ERROR'.

**5. Return Value:**

After the evolutionary loop completes (`steps` iterations), the function returns the content of the final (best) solution.

## Available agent tools

The `bp_tools.py` file provides a suite of functions and classes that can be used as tools by agents. This list details key tools and a brief description of their function:

*   `run_os_command(str_command: string, timeout: integer)`: Executes an arbitrary command in the host operating system's shell (e.g., `ls`, `cd`, `mkdir`, `pip install <package>`, `apt-get update`). Returns the standard output from the command. Use with extreme caution due to security implications.
*   `compile_and_run_pascal_code(pasfilename: string, timeout: integer)`: Compiles and executes a Free Pascal source file (`.pas`). Accepts standard Free Pascal compiler options via the `pasfilename` string. Returns the output of the compiled program.
*   `run_php_file(filename: string, timeout: integer)`: Executes a PHP script file (`.php`) using the installed PHP interpreter. Returns the standard output generated by the script.
*   `source_code_to_string(folder_name: string)`: Recursively scans a specified folder and its subfolders for common source code file types (.py, .pas, .php, .inc, .txt, .md). It reads their content and concatenates them into a single string, structured using `<file filename="...">...</file>` XML-like tags. This is invaluable for giving an agent a comprehensive view of a project's source code for documentation, analysis, or refactoring tasks.
*   `string_to_source_code(string_with_files: string, output_base_dir: string = '.', overwrite: boolean = True, verbose: boolean = False)`: Performs the inverse operation of `source_code_to_string`. It parses a structured string (like the output of `source_code_to_string`) and recreates the specified files and directory structure within the `output_base_dir`. Useful for agents generating multiple code or documentation files.
*   `pascal_interface_to_string(folder_name: string, remove_pascal_comments: boolean = False)`: Specifically scans Pascal source files in a folder and extracts only the content located within the `interface` section of units, ignoring comments and strings. The extracted content is returned in a string structured with `<pascal_interface filename="...">...</pascal_interface>` tags. Helps agents understand Pascal unit dependencies.
*   `get_pascal_interface_from_file(filename: string, remove_pascal_comments: boolean = False)`: Returns the Pascal interface section from a single Pascal source code file.
*   `get_pascal_interface_from_code(content: string, remove_pascal_comments: boolean = False)`: Extracts the interface section from Pascal source code provided as a string.
*   `remove_pascal_comments_from_string(code_string: string)`: Removes all comments from a Delphi/Pascal code string. Handles single-line comments (//), brace comments ({ }), and parenthesis-asterisk comments ((* *)). Preserves comment-like text inside string literals.
*   `save_string_to_file(content: string, filename: string)`: Writes the given string `content` to the specified `filename`. If the file exists, it is overwritten. A fundamental tool for agents to output generated text or code.
*   `append_string_to_file(content: string, filename: string)`: Appends the given string `content` to the end of the specified `filename`. Unlike `save_string_to_file`, this preserves existing file content.
*   `load_string_from_file(filename: string)`: Reads the entire content of the specified `filename` and returns it as a single string. Allows agents to read existing files.
*   `copy_file(source_filename: string, dest_filename: string)`: Copies the file located at `source_filename` to `dest_filename`. Standard file system copy operation.
*   `get_file_size(filename: string)`: Returns the size of a specified file in bytes as an integer. Useful for file management tasks.
*   `is_file(filename: string)`: Returns true if the specified path is a file. Implemented as `os.path.isfile(filename)`.
*   `force_directories(file_path: string)`: Extracts the directory path from a full file path and creates the directory structure if it does not already exist. Useful for ensuring parent directories exist before creating files.
*   `count_file_lines(filename: string)`: Returns the number of lines in a text file as an integer.
*   `get_line_from_file(file_name: string, line_number: integer)`: Reads a specified line from a text file (1-based index). Useful for finding specific lines where compilers report errors.
*   `print_file_lines(filename: string, start_line: integer, end_line: integer)`: Prints lines from `start_line` to `end_line` of the specified file. Useful in combination with `get_line_from_file` for finding bugs in source code.
*   `replace_line_in_file(file_name: string, line_number: integer, new_content: string)`: Replaces a specified line in a text file with new content. The line_number is 1-based.
*   `insert_lines_into_file(file_name: string, line_number: integer, new_content: string)`: Inserts new content before a specified line in a text file. The original line and all subsequent lines are shifted down.
*   `replace_in_file(filename: string, old_value: string, new_value: string)`: Reads the content of `filename`, replaces all occurrences of `old_value` with `new_value` in the content, and writes the modified content back to the same file. Returns the modified content string. Useful for in-place file patching.
*   `replace_in_file_from_files(filename: string, file_with_old_value: string, file_with_new_value: string)`: Reads content from `file_with_old_value` and `file_with_new_value`, then replaces all occurrences of the old content with the new content within the `filename` file. Returns the modified content string of `filename`.
*   `trim_right_lines(multi_line_string: string)`: Performs a right trim on all lines of a string, removing trailing whitespace from each line.
*   `trim_right_lines_in_file(filename: string)`: Performs a right trim on all lines of the specified file, removing trailing whitespace from each line.
*   `get_files_in_folder(folder: string = 'solutions', fileext: string = '.md')`: Returns a list of files in a folder with a given file extension. Useful for discovering files of a specific type.
*   `create_filename(topic: string, extension: string = ".md")`: Creates a filename from a topic string (unformatted) and an extension. The topic is converted to a URL-safe slug format.
*   `list_directory_tree(folder_path: string, max_depth: integer = 3, show_files: boolean = True)`: Creates a tree-like view of a directory structure. This is useful for understanding project structure without loading all file contents, saving context. Shows directories and optionally files up to a specified depth.
*   `search_in_files(folder_path: string, search_pattern: string, file_extensions: tuple = None, case_sensitive: boolean = False, max_results: integer = 50)`: Searches for a text pattern in files within a folder and its subfolders. Returns matching lines with file paths and line numbers. Much more efficient than loading all files when you need to find specific code patterns.
*   `read_file_range(filename: string, start_byte: integer, end_byte: integer)`: Reads a specific byte range from a file. This is useful for very large files where you only need to inspect a portion, saving memory and context.
*   `get_file_info(filepath: string)`: Gets metadata about a file without reading its content. Returns a dictionary containing file properties (size, modified_time, is_file, is_dir, exists, readable, writable). Efficient for checking file properties before deciding whether to load the full content.
*   `list_directory(folder_path: string, pattern: string = "*", recursive: boolean = False, files_only: boolean = False, dirs_only: boolean = False)`: Lists files and directories in a folder with optional filtering. More flexible than `get_files_in_folder` with glob pattern matching support. Can search recursively and filter by type.
*   `mkdir(directory_path: string, parents: boolean = True)`: Creates a directory. If `parents=True`, creates intermediate directories as needed (similar to `mkdir -p` in Unix).
*   `extract_function_signatures(filename: string, language: string = "python")`: Extracts function and class signatures from a source code file without loading the full implementation. Helps understand code structure efficiently. Currently supports Python, JavaScript, Java, and PHP.
*   `compare_files(file1: string, file2: string, context_lines: integer = 3)`: Compares two files and shows the differences in a unified diff format. Useful for understanding what changed between versions. Returns a diff output with configurable context lines.
*   `delete_file(filepath: string)`: Deletes a file from the filesystem. Returns `True` if successful. Raises appropriate exceptions if the file doesn't exist or is a directory.
*   `delete_directory(directory_path: string, recursive: boolean = False)`: Deletes a directory. If `recursive=True`, deletes the directory and all its contents. Use with caution.
*   `count_lines_of_code(folder_path: string, file_extensions: tuple = ('.py', '.js', '.java', '.cpp', '.c', '.php', '.rb'))`: Counts lines of code in a project, broken down by file type. Helps understand project size and composition without loading all files. Returns a dictionary with file extensions as keys and line counts as values.
*   `read_first_n_lines(filename: string, n: integer)`: Reads the first `n` lines of a file. Useful for previewing large files without loading everything into memory. Returns the first `n` lines as a string.
*   `read_last_n_lines(filename: string, n: integer)`: Reads the last `n` lines of a file. Useful for reading log files or checking the end of large files. Returns the last `n` lines as a string.
*   `delete_lines_from_file(filename: string, start_line: integer, end_line: integer = None)`: Deletes specific lines from a file. If `end_line` is `None`, only deletes the `start_line`. Both `start_line` and `end_line` are 1-based indices (inclusive). Returns the updated file content as a string.
*   `load_image(filepath: string)`: Loads an image file (PNG, JPG, BMP, GIF, etc.) into the agent's visual context. The image appears in the next turn so the agent can reason about its contents. Supports multiple images per step. Always available (no flags needed). Useful for inspecting matplotlib plots, generated diagrams, screenshots, or any visual output.
*   `diff_images(image1_path: string, image2_path: string, output_path: string = None, mode: string = "highlight")`: Visually compares two images and produces a diff image highlighting the differences. Modes: `highlight` (red overlay on changed pixels) or `side_by_side` (before/diff/after). Returns the diff image path and a percentage of changed pixels. Use `load_image` on the result to view.
*   `screen_ocr(image_path: string, region: string = None, language: string = "eng")`: Extracts text from an image using OCR (Tesseract). Optionally crop to a region (`x,y,width,height`) before OCR. Requires `tesseract-ocr` to be installed (`sudo apt install tesseract-ocr`).
*   `canvas_create(width: integer, height: integer, output_path: string, bg_color: string = "white")`: Creates a blank canvas image of the specified size and background color. Use `canvas_draw` to add shapes and `load_image` to view.
*   `canvas_draw(image_path: string, shape: string, coords: string, color: string = "red", fill: string = None, line_width: integer = 2, text: string = None, font_size: integer = 16)`: Draws shapes and text on any image file. Supported shapes: `rect`, `circle`, `ellipse`, `line`, `arrow`, `text`. Works on canvases, screenshots, photos — any image. Call multiple times to build up a drawing.

### Sub-assistant Tool Classes

In addition to the function-based tools above, `bp_tools.py` provides several Tool classes that wrap agents as tools, allowing them to be used by other agents:

*   `Summarize(agent)`: A sub-assistant that returns a summary of a provided string.
*   `SummarizeUrl(agent)`: A sub-assistant that returns a summary of a web page given its URL.
*   `SummarizeLocalFile(agent)`: A sub-assistant that returns a summary of a local file.
*   `Subassistant(agent)`: A general-purpose sub-assistant similar in capability to the main agent. Can be used to delegate tasks.
*   `InternetSearchSubassistant(agent)`: A sub-assistant dedicated to internet searches. Useful for delegating research tasks.
*   `CoderSubassistant(agent)`: A sub-assistant specialized in coding tasks.
*   `GetRelevantInfoFromFile(agent)`: A sub-assistant that extracts relevant information about a specific topic from a local file.
*   `GetRelevantInfoFromUrl(agent)`: A sub-assistant that extracts relevant information about a specific topic from a URL.

All sub-assistant classes support a `restart_chat` parameter to control whether the sub-assistant should maintain context from previous interactions or start fresh.

## Create a team of agents (sub-assistants) and use them as tools

### Core Concepts

Beyond Python Smolagents is built around the concept of AI agents equipped with tools to interact with their environment and solve tasks.

*   **Agents** (inherited from [smolagents](https://github.com/huggingface/smolagents)): Autonomous entities powered by language models that receive instructions and use available tools to achieve objectives. Different agent types (`CodeAgent`, `ToolCallingAgent`, `MultiStepAgent`) are available, each tailored for potentially different purposes and capable of being configured with specific tool sets:
    *   `CodeAgent`: Specialized in code generation, execution, and debugging across multiple languages.
    *   `ToolCallingAgent`: A general-purpose agent capable of utilizing a defined set of tools.
    *   `MultiStepAgent`: Designed to break down complex tasks into smaller steps and execute them sequentially or iteratively.
*   **Models** (inherited from [smolagents](https://github.com/huggingface/smolagents)): The underlying Language Models (LLMs) that provide the cognitive capabilities for the agents, enabling them to understand tasks, reason, and generate responses or code. The framework integrates with various LLMs via the LiteLLM library, allowing users to select models based on cost, performance, context window size, and specific capabilities.
*   **Tools:** (inherited from [smolagents](https://github.com/huggingface/smolagents)): Functions or utilities that agents can call to perform actions in the environment. These abstract interactions such as running OS commands, accessing the filesystem, interacting with the internet, or executing code in different programming languages. Tools are fundamental; without them, agents can only generate text; with them, they can *act*. The framework provides many built-in tools, and users can define custom ones.
*   **Sub-assistants:** Instances of agents are treated as tools and provided to a primary agent (often called "the boss"). This allows a higher-level agent to delegate specific sub-tasks to specialized agents. For example, a main agent tasked with building a project might delegate code generation to a `CoderSubassistant` or research to an `InternetSearchSubassistant`. This enables building complex, modular artificial workforce and leverages the specialized capabilities of different agent configurations.
*   **Base Tools (`add_base_tools=True/False`)** (inherited from [smolagents](https://github.com/huggingface/smolagents)): A crucial parameter when initializing agents. It controls whether an agent automatically receives a default, standard set of tools provided by the Beyond Python Smolagents framework.
    *   Setting `add_base_tools=True` equips the agent with a common set of utilities right out of the box. This set typically includes tools for basic file operations (`save_string_to_file`, `load_string_from_file`), web interaction (`VisitWebpageTool`, `DuckDuckGoSearchTool`), and Python execution (`PythonInterpreterTool`), among others. These are added *in addition to* any tools explicitly provided in the `tools` list during initialization. This is useful for creating general-purpose agents.
    *   Setting `add_base_tools=False` means the agent will *only* have access to the tools explicitly passed to it via the `tools` parameter during initialization. This allows for creating highly minimal or very specifically-purposed agents with a restricted set of actions, which can be beneficial for security or task focus.
*   **bp_tools.py**: The module containing Beyond Python Smolagents-specific tools that extend the base smolagents functionality. These tools enable agents to interact with the filesystem, compile and run code in multiple languages, and delegate tasks to specialized sub-assistants.
    *   **File system utilities**: `save_string_to_file`, `load_string_from_file`, `copy_file`, `get_file_size`, `is_file`, `mkdir`, `delete_file`, `delete_directory`.
    *   **Source code handling**: `source_code_to_string`, `string_to_source_code`, `extract_function_signatures`, `list_directory_tree`, `search_in_files`.
    *   **OS command execution**: `run_os_command` for running arbitrary shell commands.
    *   **Sub-assistant tool classes**: `Summarize`, `SummarizeUrl`, `SummarizeLocalFile`, `Subassistant`, `CoderSubassistant`, `InternetSearchSubassistant`, `GetRelevantInfoFromFile`, `GetRelevantInfoFromUrl`.

### Creating the team
Beyond Python Smolagents allows you to compose complex working groups by having agents delegate tasks to other specialized agents, referred to as sub-assistants. This modular approach helps manage complexity and leverage agents optimized for specific tasks (e.g., coding, internet search, summarization).

The library provides wrapper classes (Subassistant, CoderSubassistant, InternetSearchSubassistant, Summarize, etc.) that turn an agent instance into a tool that another agent can call.

Here's an example demonstrating how to set up a "boss" agent that can utilize other agents as sub-assistants:
```
no_tool_agent = ToolCallingAgent(tools=[], model=model, add_base_tools=False)
tooled_agent = ToolCallingAgent(tools=tools, model=model, add_base_tools=True)
internet_search_agent = ToolCallingAgent(tools=[save_string_to_file, load_string_from_file], model=model, add_base_tools=True)

subassistant = Subassistant(tooled_agent)
internet_search_subassistant = InternetSearchSubassistant(internet_search_agent)
coder_subassistant = CoderSubassistant(coder_agent)
summarize = Summarize(no_tool_agent)
summarize_url = SummarizeUrl(no_tool_agent)
summarize_local_file = SummarizeLocalFile(no_tool_agent)
get_relevant_info_from_file = GetRelevantInfoFromFile(no_tool_agent)
get_relevant_info_from_url = GetRelevantInfoFromUrl(no_tool_agent)

tools = [save_string_to_file, load_string_from_file, copy_file, get_file_size,
  source_code_to_string, string_to_source_code, pascal_interface_to_string,
  replace_in_file, replace_in_file_from_files,
  subassistant, coder_subassistant, internet_search_subassistant,
  summarize, summarize_url, summarize_local_file,
  get_relevant_info_from_file, get_relevant_info_from_url,
  run_os_command, run_php_file, compile_and_run_pascal_code,
  ]

task_str="""Code, test and debug something that will impress me!
For completing the task, you will first plan for it.
You will decide what task will be assigned to each of your sub-assistants.
You will decide the need for researching using internet_search_subassistant before you actually start coding a solution."""

the_boss = CodeAgent(model=coder_model, tools = tools, add_base_tools=True)
the_boss.run(task_str)
```
***
### 🔥🚨 EXTREME SECURITY RISK 🚨🔥
***

**This implementation grants agents extensive access and control over the environment in which they run.** This level of control is intentionally designed to enable powerful automation and interaction capabilities across different languages and the operating system (including file system access, running arbitrary OS commands, and executing code in various languages).

**CONSEQUENTLY, USING THIS SOFTWARE IN AN ENVIRONMENT CONTAINING SENSITIVE DATA, PRODUCTION SYSTEMS, OR IMPORTANT PERSONAL INFORMATION IS HIGHLY DANGEROUS AND STRONGLY DISCOURAGED.**

**YOU MUST ONLY RUN THIS CODE INSIDE A SECURELY ISOLATED ENVIRONMENT** specifically set up for this purpose, such as:
*   **A dedicated Virtual Machine (VM):** Configure a VM with minimal or no sensitive data, isolated from your main network if possible. Treat anything inside the VM as potentially compromised.
*   **A locked-down Container (like Docker):** Use containerization to create an isolated filesystem and process space. Ensure no sensitive volumes from your host machine are mounted into the container. Limit network access if possible.

**DO NOT** run this code directly on your primary development machine, production servers, personal computer, or any environment with valuable data or system access you wish to protect.

**USE THIS SOFTWARE ENTIRELY AT YOUR OWN RISK! The developers explicitly disclaim responsibility for any damage, data loss, security breaches, or other negative consequences resulting from the use of this software in an insecure or inappropriate environment.** This warning cannot be overstated.
***

## Other Publications from the Author of this Fork

Optimizing the first layers of a convolutional neural network:
- [Color-aware two-branch DCNN for efficient plant disease classification](https://www.researchgate.net/publication/361511874_Color-Aware_Two-Branch_DCNN_for_Efficient_Plant_Disease_Classification).
- [Reliable Deep Learning Plant Leaf Disease Classification Based on Light-Chroma Separated Branches](https://www.researchgate.net/publication/355215213_Reliable_Deep_Learning_Plant_Leaf_Disease_Classification_Based_on_Light-Chroma_Separated_Branches)

Optimizing deep layers of a convolutional neural network:
- [Grouped Pointwise Convolutions Reduce Parameters in Convolutional Neural Networks](https://www.researchgate.net/publication/360226228_Grouped_Pointwise_Convolutions_Reduce_Parameters_in_Convolutional_Neural_Networks)
- [An Enhanced Scheme for Reducing the Complexity of Pointwise Convolutions in CNNs for Image Classification Based on Interleaved Grouped Filters without Divisibility Constraints](https://www.researchgate.net/publication/363413038_An_Enhanced_Scheme_for_Reducing_the_Complexity_of_Pointwise_Convolutions_in_CNNs_for_Image_Classification_Based_on_Interleaved_Grouped_Filters_without_Divisibility_Constraints)

Optimizing LLMs:
- [Saving 77\% of the Parameters in Large Language Models Technical Report](https://www.researchgate.net/publication/388835829_SAVING_77_OF_THE_PARAMETERS_IN_LARGE_LANGUAGE_MODELS_TECHNICAL_REPORT)

