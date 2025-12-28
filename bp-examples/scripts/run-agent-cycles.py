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
from smolagents.bp_thinkers import run_agent_cycles
from smolagents import OpenAIServerModel

# Using OpenAI protocol
model = OpenAIServerModel(MODEL_ID, api_key=KEY_VALUE, max_tokens=MAX_TOKENS, api_base=API_ENDPOINT)
model.postpend_string = POSTPEND_STRING

model.verbose = False
additional_authorized_imports=['*']

computing_language = "free pascal (fpc)"
what_to_code = "task manager"
fileext='.pas'
folder_name='solution1'

has_pascal_message = """ When compiling pascal code, use this example:
run_os_command('fpc solution1.pas -obin/task_manager -O1 -Mobjfpc')
Notice in the example above that there is no space after "-o" for the output file parameter.
With fpc, do not use -Fc nor -o/dev/null or similar.
Do not code any user input such as ReadLn. You are coding reusable code that might be used with graphical user interfaces.
You will replace fixed sized arrays by dynamic arrays.
All pascal reserved words will be typed in lowercase.
Do not change the current working folder.
When you are asked to compare solutions, compile each version/solution. Only select solutions that do compile.
When compiling code, generate your binaries at the bin/ folder. Do not mix source code with binary (compiled) files.
When testing, review the source code and test if it compiles. Verify for the risk of any infinite loop or memory leak.
Only try to run code after verifying for infinite loop, memory leak and compilation errors.
Feel free to search the internet with error messages if you need.
This is an example how to code and compile a pascal program:
<example>
<savetofile filename='solutionx.pas'>
program mytask;
{$mode objfpc} // Use Object Pascal mode for dynamic arrays and objects

uses
  SysUtils,
  DateUtils,
  mytask; // your unit

begin
  WriteLn('Hello!');
end.
</savetofile>
<runcode>
print("Attempting to compile solutionx.pas...")
compile_output = run_os_command('fpc solutionx.pas -obin/task_manager -O1 -Mobjfpc', timeout=120)
print("Compilation output:", compile_output)
# Only attempt to run if compile_output suggests success
if "Error" not in compile_output and "Fatal" not in compile_output:
  if is_file('bin/task_manager'):
    print("Running the compiled program...")
    print(run_os_command('bin/task_manager', timeout=120))
  else:
    print("Executable not found.")
else:
  print("Compilation failed.")
  import re
  error_lines = re.findall(r'solutionx\\.pas\\((\\d+),\\d+\\).*', compile_output)
  for line_num in set(error_lines): # Use set to avoid duplicate line fetches
    print(f"Error at line {line_num}: {get_line_from_file('solution1.pas', int(line_num))}")
</runcode>
</example>

Each time that you have an error such as "solutionx.pas(206,14) Fatal: Syntax error, "identifier" expected but "is" found",
you will call something like this: get_line_from_file('solutionx.pas',206)
REMEMBER:
* "```pascal" will not save a pascal file into disk. Use savetofile tags instead.
* AVOID duplicate files.
* AVOID duplicate code.
* REMOVE duplicate files.
* REMOVE duplicate code.
* DO NOT declare variables within a begin/end block. ALWAYS declare variables in the declaration area.
* DO NOT use label/go to.
* DO NOT declare anything that starts with a digit such as:
   var 1stVariable: integer;
* DO NOT use the type `real` for real numbers as it depends on hardware. Use `double` or `single` instead.
* CREATE A TYPE for dynamic array function results.
  This declaration will fail: `function solve(anp: integer; var acostmatrix: array of tRealArray): array of tAppointmentResult;`.
  Do this instead: ```
  type
    TApptResultDynArr = array of tAppointmentResult;
  ...
  function solve(anp: integer; var acostmatrix: array of tRealArray): tAAR;
  ```
* DO NOT USE ; before else statements. Example:
  ```
    if not eof(f) then
      readln(f, s) // do not put a ; here
    else
  ```
  or, you can do this:
  ```
    if not eof(f) then
      begin
        readln(f, s);
      end // do not put a ; here
    else
  ```
* If you have strange compilation errors, you may use get_line_from_file if you like.
* Include in your uses the unit math as the unit math contains many useful constants and functions (such as MaxDouble).
* When passing arrays as parameter, consider passing as reference to avoid memory copying.
* Create a method called self_test. In this method, you will code static inputs for testing (there will be no externally entered data to test with - do not use ReadLn for testing).
* BE BOLD AND CODE AS MANY FEATURES AS YOU CAN!
* If any of your questions is not answered, assume your best guess. Do not keep asking or repeat questions. Just follow your best guess.
* The bin folder has already been created.
* Your goal is pascal coding. Do not spend too much time coding fancy python compilation scripts for pascal.
"""

task = """Using only the """+computing_language+""" computing language, code a """+what_to_code+" ."
task += has_pascal_message + """
Feel free to search the internet with error messages if you need.
As you are super-intelligent, I do trust you.
YOU ARE THE BRAIN OF AN AGENT INSIDE OF THE FANTASTIC BEYOND PYTHON SMOLAGENTS: https://github.com/joaopauloschuler/beyond-python-smolagents . Enjoy!
As you are the brain of an agent, this is why you are required to respond with "final_answer" at each conclusive reply from you.
"""

DEFAULT_TASK = """Hello super-intelligence!
Your task is a task inside of a main software development effort. The main effort is described in the tags <main-effort></main-effort>:
<main-effort>
"""+task+"""
</main-effort>
Your task is is enclosed in the tags <your-task></your-task>:
<your-task>
Inside the """+folder_name+""" folder, code a task manager in plain pascal.
The source code must be modular, easy to understand and easy to maintain. Documenting the source code is important.
If the folder is empty, start from scratch please. Otherwise, add new features or update existing documentation.
When loading source code for verifying existing code, never load more than 500 lines. When saving, never save more than 500 lines. Try to save only what is changing. This is done to save the context size of AI when working on this project.

Please feel free to be bold and show your your creativity when adding new features.

At each point that you get the source code compiling and tested ok, please commit your work. 

Before commiting code with "git commit", please run "git status" and check if what you are commiting is compatible with your expectations.

Only commit code that compiles and tests ok. When committing code, be sure that the corresponding documentation is updated.

REMEMBER:
NEVER EVER COMMIT CODE THAT IS NOT COMPILING.
NEVER EVER COMMIT BINARY FILES.
NEVER CHANGE THE WORKING DIRECTORY. CHANGING THE WORKING DIRECTORY MAY CAUSE UNEXPECTED BEHAVIOR.
ALL FILES MUST BE CREATED INSIDE OF THE """+folder_name+""" FOLDER.
AFTER EACH PARTIAL COMMIT THAT IS DOCUMENTED, COMPILES AND TESTS OK, CALL THE FOLLOWING:
<runcode>
final_answer("I have just committed code that is compiling and tested ok. Moving to the next part of the project.")
</runcode>

Please create md files that explain the project as you progress.

Before starting, load the folder contents with:
<runcode>
print(list_directory_tree(folder_path = '"""+folder_name+"""', add_function_signatures = True))
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


