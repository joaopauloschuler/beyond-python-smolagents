# Get mandatory KEY_VALUE from user
import getpass
KEY_VALUE = getpass.getpass("Enter your API key (mandatory): ").strip()
while not KEY_VALUE:
    print("API key is required!")
    KEY_VALUE = getpass.getpass("Enter your API key (mandatory): ").strip()

# Get optional parameters with defaults
api_endpoint_input = input("Enter API endpoint (press Enter for default: https://api.poe.com/v1): ").strip()
API_ENDPOINT = api_endpoint_input if api_endpoint_input else "https://api.poe.com/v1"

model_id_input = input("Enter model ID (press Enter for default: Claude-Haiku-4.5): ").strip()
MODEL_ID = model_id_input if model_id_input else "Claude-Haiku-4.5"

cycles_cnt_input = input("Enter number of cycles (press Enter for default: 1): ").strip()
if cycles_cnt_input:
    try:
        CYCLES_CNT = int(cycles_cnt_input)
    except ValueError:
        print(f"Invalid number '{cycles_cnt_input}', using default: 1")
        CYCLES_CNT = 1
else:
    CYCLES_CNT = 1

max_steps_input = input("Enter max steps per cycle (press Enter for default: 100): ").strip()
if max_steps_input:
    try:
        MAX_STEPS_PER_CYCLE = int(max_steps_input)
    except ValueError:
        print(f"Invalid number '{max_steps_input}', using default: 100")
        MAX_STEPS_PER_CYCLE = 100
else:
    MAX_STEPS_PER_CYCLE = 100

planning_interval_input = input("Enter planning interval (press Enter for default: 22): ").strip()
if planning_interval_input:
    try:
        PLANNING_INTERVAL = int(planning_interval_input)
    except ValueError:
        print(f"Invalid number '{planning_interval_input}', using default: 22")
        PLANNING_INTERVAL = 22
else:
    PLANNING_INTERVAL = 22

task_file_input = input("Enter task description file path (press Enter to use default task): ").strip()

POSTPEND_GEMINI_FLASH_VIA_POE = ''
POSTPEND_GEMINI_FLASH_VIA_GOOGLE = ''
POSTPEND_CLAUDE = ''
POSTPEND_STRING = POSTPEND_GEMINI_FLASH_VIA_POE
GLOBAL_EXECUTOR = 'exec'
MAX_TOKENS = 64000

import smolagents
from smolagents.bp_tools import *
from smolagents.bp_utils import *
from smolagents.bp_thinkers import *
from smolagents import OpenAIServerModel

# Using OpenAI protocol
model = OpenAIServerModel(MODEL_ID, api_key=KEY_VALUE, max_tokens=MAX_TOKENS, api_base=API_ENDPOINT)
model.postpend_string = POSTPEND_STRING

model.verbose = False
additional_authorized_imports=['*']
folder_name='solution1'

DEFAULT_TASK = """Hello super-intelligence!
Your task is a task inside of a main software development effort. The main effort is described in the tags <main-effort></main-effort>:
<main-effort>
Document the source code found in the """+folder_name+""" folder using markdown.
The project should have only one markdown file: the main readme.md file that documents the entire project.
In the case that there are other markdown files, you will merge these files into the main readme.md file avoiding duplication of information.
After mergind, delete the other markdown files.
</main-effort>
Your task is enclosed in the tags <your-task></your-task>:
<your-task>
Before starting, load the folder contents with:
<runcode>
print(list_directory_tree(folder_path = '"""+folder_name+"""', add_function_signatures = True))
</runcode>
You will first pick a portion of the documentation that you consider that is either missing, disrespecting these instructions or could be improved.
Then, edit the markdown documentation for the source code found in the """+folder_name+""" folder.
Use may use small code snippets where appropriate to illustrate key points.
Ensure that the documentation is clear, concise, and easy to understand for future developers who may work with this code.
Each time you finish a portion of the documentation, git commit your changes. Do not try to push.
AFTER EACH PARTIAL COMMIT, CALL THE FOLLOWING:
<runcode>
final_answer("I have just committed code that is compiling and tested ok. Moving to the next part of the project.")
</runcode>
</your-task>
May the force be with you. I do trust your judgement."""

# Load task from file if provided, otherwise use DEFAULT_TASK
task_description = DEFAULT_TASK
if task_file_input:
    try:
        with open(task_file_input, 'r', encoding='utf-8') as f:
            task_description = f.read()
        print(f"Task description loaded from: {task_file_input}")
    except FileNotFoundError:
        print(f"Warning: File '{task_file_input}' not found. Using default task.")
    except Exception as e:
        print(f"Warning: Error reading file '{task_file_input}': {e}. Using default task.")

run_agent_cycles(model=model,
    task_str=task_description,
    cycles_cnt=CYCLES_CNT,
    planning_interval=PLANNING_INTERVAL,
    max_steps=MAX_STEPS_PER_CYCLE)
