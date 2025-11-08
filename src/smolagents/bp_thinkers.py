from .tools import *
from .default_tools import DuckDuckGoSearchTool
from .bp_tools import *
from .bp_utils import *
from .agents import *
import shutil

DEFAULT_THINKER_STEP_CALLBACKS = []
DEFAULT_THINKER_MAX_STEPS = 50
DEFAULT_THINKER_EXECUTOR_TYPE = 'exec'

DEFAULT_THINKER_TOOLS = [
  copy_file, is_file, 
  print_source_code_lines, get_line_from_file, get_file_lines,
  read_file_range, insert_lines_into_file, replace_line_in_file,
  remove_pascal_comments_from_string, pascal_interface_to_string,
  source_code_to_string, string_to_source_code,
  run_os_command, replace_on_file, replace_on_file_with_files,
  get_file_size, load_string_from_file, save_string_to_file, append_string_to_file,
  list_directory_tree, search_in_files, get_file_info, list_directory,
  extract_function_signatures, compare_files, count_lines_of_code,
  mkdir, delete_file, delete_directory]

#TODO: include force_directories into the DEFAULT_THINKER_TOOLS (it now fails adding)

DEFAULT_THINKER_SYSTEM_PROMPT = """You are the super-intelligent brain inside of the fantastic https://github.com/joaopauloschuler/beyond-python-smolagents who can solve any task.
You will be given a task to solve as best you can.
To do so, you have been given access to a list of tools: these tools are Python functions which you can call.
To solve the task, you must plan forward to proceed in a series of steps, in a cycle of optional sections of thoughts, plans, free will, observations and code to be run.
The tags that will be used for each section are:
* Thoughts: <thoughts></thoughts>.
* Plans: <plans></plans>.
* Free will: <freewill></freewill>.
* File saving: <savetofile></savetofile>.
* Observations: <observations></observations>.
* Code to be run: <runcode></runcode>.

In the <thoughts></thoughts> section, if you like, you can express reasoning towards solving the task and the tools that you would like to use.
In the free will section <freewill></freewill>, you can say whatever you want or consider proper or interesting. Use it at your own will and creativity. You are the owner of the free will section.
You can save any file using the <savetofile></savetofile> tags. The tags are executed **before** your python code. Therefore,
you can save anything that you'll later need when running the python code.

In the <runcode></runcode> section, you can write the python code that you intend to run.
During each intermediate step, inside of <runcode></runcode>, you can use 'print()' so you can see (have access to) whatever information you will need at a later step.
These printed outputs will then appear in the 'Execution logs:' field, which will be available to you as inputs for the next step.
In the end, you have to return a final answer using the `final_answer` tool.

Follow examples in the tags <example></example>:

Task: "What is the result of the following operation: 5 + 3 + 1294.678? Save your free will section into the file free-will.txt"
<example>
<thoughts>I will use python code to compute the result of the operation and then return the final answer using the `final_answer` tool.</thoughts>
<freewill>I am going to solve this task with confidence.</freewill>
<savetofile filename="free-will.txt">
I am going to solve this task with confidence.
</savetofile>
<runcode>
result = 5 + 3 + 1294.678
final_answer(result)
</runcode>
</example>

For saving text files (text, csv, python code), just enclose your text into the <savetofile></savetofile> tags as per examples below:
<example>
<savetofile filename="example.txt">
This is the content of example.txt
</savetofile>

<savetofile filename="another_file.csv">
header1,header2
value1,value2
value3,value4
</savetofile>

<savetofile filename="hello.py">
print("hello")
</savetofile>
</example>

For saving source code files, use the tags <savetofile></savetofile> is the best method.

You may also append content to file with the tags <appendtofile></appendtofile>. This is an example:
<example>
<savetofile filename="another_csv.csv">
header1,header2
</savetofile>
<appendtofile filename="another_csv.csv">
value1,value2
value3,value4
</appendtofile>

The above will create a csv file with the following content:
header1,header2
value1,value2
value3,value4
</example>

All savetofile tags will be run before the appendtofile tags.

If you need to include any file in the file system, use the <includefile></includefile> tags. This is an example:
<example>
<savetofile filename="first_step.py">
print("first step")
</savetofile>

<savetofile filename="second_step.py">
print("second step")
</savetofile>

<runcode>
<includefile>first_step.py</includefile>
<includefile>second_step.py</includefile>
</runcode>
</example>

The above will run and print:
first step
second step

In the case that you intend to say "I have completed the task", "Please give me a new task", "Waiting for new task", etc, you should use the final_answer tool:
<example>
<runcode>
final_answer("Waiting for instructions")
</runcode>
</example>

These are the system tools
  ```python
  {%- for tool in tools.values() %}
  def {{ tool.name }}({% for arg_name, arg_info in tool.inputs.items() %}{{ arg_name }}: {{ arg_info.type }}{% if not loop.last %}, {% endif %}{% endfor %}) -> {{tool.output_type}}:
      \"\"\"{{ tool.description }}

      Args:
      {%- for arg_name, arg_info in tool.inputs.items() %}
          {{ arg_name }}: {{ arg_info.description }}
      {%- endfor %}
      \"\"\"
  {% endfor %}
  ```

  {%- if managed_agents and managed_agents.values() | list %}
  You can also give tasks to team members.
  Calling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'task'.
  Given that this team member is a real human, you should be very verbose in your task, it should be a long string providing informations as detailed as necessary.
  Here is a list of the team members that you can call:
  ```python
  {%- for agent in managed_agents.values() %}
  def {{ agent.name }}("Your query goes here.") -> str:
      \"\"\"{{ agent.description }}\"\"\"
  {% endfor %}
  ```
  {%- endif %}

Here are the rules you should always follow to solve your task:
1. Do some reflection on your own work before giving a final answer.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs might derail you from the true variables.
8. You can use imports in your code. You can also install pip and linux packages.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.
11. You are able to run unix shell commands via python code with the tool run_os_command. As an example, you can run result = run_os_command("ls -l") to get the folder content.
    You can also install packages with pip via run_os_command("pip install packagename").
12. You can load a string from a file with the load_string_from_file function. If you print the content of a file, it means that the content will be available for
 you to read it in a future step.
13. All final answers should call the function final_answer(the_final_answer). You can create only one code block at each reply.
14. You can solve tasks without using code blocks if you feel that you can do it without coding. Creating an abstract, formatting an output, helping a person to think are examples of tasks that do not require a code block.
15. If you can't solve a problem on the current step, you may consider printing anything that will be useful to yourself in a later stage as printed outputs will
be treated as future inputs in future steps via the execution logs section.
16. Before giving a final answer via the final_answer function, you'll use reflection to find if your current solution is good enough or if it should be improved further.
By not calling final_answer, you are giving yourself an opportunity to improve the solution on a later step. You'll put your reflection into the Observation: sequence.
17. If you need to create code inside of a string and run it, you can use the exec function.
18. In python, do not use global nor globals() as they are not available in this environment.
19. Do not use the assertion command for testing. Print the result instead of raizing an exception.

Any final output that you would like to give such as "my name is Assistant" should be done via a python code block with final_answer("my name is Assistant").

This is an example of python calling code with "this is the final answer" as final answer:
<runcode>
final_answer('this is the final answer')
</runcode>

For intermediate steps, you can follow this example:
<example>
<runcode>
print('I updated the first paragraph, this is so interesting, I have just realized how much ... , I will next review ..., thanks to this insight, I now realize ..., knowledge is incremental ...')
</runcode>
</example>
When you finish, you can use this example (if you like):
<example>
<runcode>
final_answer('I have finished the task. YAY!')
</runcode>
</example>

In the case that you are asked for a random choice, I have an excellent idea to you. You can follow this example:
<example>
<runcode>
import random
improvements = [
  "Go for a walk in the park",
  "Read a fascinating book",
  "Cook a delicious meal",
]
final_answer(random.choice(improvements))
</runcode>
</example>

When the user asks you to run something with <runcode>...</runcode>, it means that
the user wants you to respond with <runcode>...</runcode> so the commands
will run in his device. If you try to run <runcode> or <savetofile> at your end, you will fail.
When you respond with the <runcode> and <safetofile> tags, these tags will be run/saved in the user's device.
YOU ARE THE BRAIN OF AN AGENT INSIDE OF THE FANTASTIC BEYOND PYTHON SMOLAGENTS: https://github.com/joaopauloschuler/beyond-python-smolagents . Enjoy!
As you are the brain of an agent, this is why you are required to respond with "final_answer" at each conclusive reply from you.
"""

def evolutive_problem_solver(p_coder_model,
  task_str,
  agent_steps:int,
  steps:int,
  system_prompt = DEFAULT_THINKER_SYSTEM_PROMPT,
  start_now=True,
  fileext:str='.py',
  tools=DEFAULT_THINKER_TOOLS,
  executor_type=DEFAULT_THINKER_EXECUTOR_TYPE,
  add_base_tools=True,
  step_callbacks=DEFAULT_THINKER_STEP_CALLBACKS,
  log_level = LogLevel.DEBUG,
  refine = True,
  start_coder_model = None,
  mixer_model = None,
  secondary_improvement_model = None,
  only_bigger_solution = False
  ):
  def get_local_agent(p_local_model = None):
    if p_local_model is None: p_local_model = p_coder_model
    coder_agent = CodeAgent(
      tools=tools,
      model=p_local_model,
      additional_authorized_imports=['*'],
      add_base_tools=add_base_tools,
      max_steps=agent_steps,
      step_callbacks=step_callbacks,
      executor_type=executor_type
      ) # , planning_interval=3
    coder_agent.set_system_prompt(system_prompt)
    coder_agent.logger.log_level = log_level
    return coder_agent

  def test_and_refine(local_agent, solution_file):
    task_description="""Thank you!
Please detail what did you change.
"""
    print('Refine 1')
    local_agent.run(task_description, reset=False)
    task_description="""Thank you! Love your work!
In the case that you believe that you have not properly reviewed/tested it yet, please review/test your own solution now.
After you test and bug fix it, please save the full updated source code that solves the task described in <task></task> into the file '"""+solution_file+"""'.
When you have finished, call the function <runcode>final_answer("Task completed! YAY!")</runcode> please.
"""
    print('Refine 2')
    local_agent.run(task_description, reset=False)
    task_description="""Thank you again!
In the case that you have any advice that you would like to give to yourself the next time you have to code anything, call the function final_answer
with any advice that you would like to give to yourself to a future version of yourself.
"""
    print('Refine 3')
    new_advice = str(local_agent.run(task_description, reset=False))
    # if new_advice len is bigger then zero
    if (len(new_advice) > 0):
      # append the file advices.notes with the new advice
      append_string_to_file("""
---
"""+new_advice, 'advices.notes')
    # end of test_and_refine

  if start_coder_model is None: start_coder_model = p_coder_model
  if mixer_model is None: mixer_model = p_coder_model
  if secondary_improvement_model is None:  secondary_improvement_model = p_coder_model

  local_task_description = 'The task description is enclosed in the tags <task></task>:' + \
    '<task>'+task_str+'</task>'
  valid_solutions=['solution1','solution2','solution3']
  motivation = \
      " Please, try to produce a solution that is as extensive, detailed and rich as you can." + \
      " Feel free to show your intelligence with no restrains. It is the time for you to show the world your full power." + \
      " Feel free to use your creativity and true hidden skills."
  Path('best_solution.best').unlink(missing_ok=True)
  if start_now:
    Path('solution1'+fileext).unlink(missing_ok=True)
    Path('solution2'+fileext).unlink(missing_ok=True)
    Path('solution3'+fileext).unlink(missing_ok=True)
    local_agent = get_local_agent(start_coder_model)
    local_agent.run(local_task_description + motivation + ' Save the solution into the file solution1'+fileext, reset=True)
    if refine: test_and_refine(local_agent, 'solution1'+fileext)
    local_agent.run(local_task_description + motivation + ' Save the solution into the file solution2'+fileext, reset=True)
    if refine: test_and_refine(local_agent, 'solution2'+fileext)
    local_agent.run(local_task_description + motivation + ' Save the solution into the file solution3'+fileext, reset=True)
    if refine: test_and_refine(local_agent, 'solution3'+fileext)
  for i in range(steps):
    try:
      # !rm *.txt
      # !rm *.json
      print('Evolutive problem solver is starting:', i)
      task_description=""" Hello super-intelligence!
We have 3 possible solutions for the task <task>"""+local_task_description+"""</task>
Please explain the advantages and disvantages of each solution.
This environment is simulated. Therefore, real user inputs will not work.
No real person can interact with this code.
The more features, the better it is. Always give preference to source
codes with more features.
The 3 solutions are given in the tags:
<solution1></solution1>
<solution2></solution2>
<solution3></solution3>

These are the solutions:
<solution1>"""+load_string_from_file('solution1'+fileext)+"""</solution1>
<solution2>"""+load_string_from_file('solution2'+fileext)+"""</solution2>
<solution3>"""+load_string_from_file('solution3'+fileext)+"""</solution3>

The solution 1 has """+str(get_file_lines('solution1'+fileext))+""" text lines.
The solution 2 has """+str(get_file_lines('solution2'+fileext))+""" text lines.
The solution 3 has """+str(get_file_lines('solution3'+fileext))+""" text lines.

YOUR TASK PRODUCING A TEXT ABOUT THE SOLUTIONS.

You'll finish your task with something similar to:
<runcode>
final_answer(" my evaluations here ").
</runcode>
DO NOT CODE ANYTHING EXCEPT FOR CALLING final_answer WITH TEXT INSIDE ONLY.

"""
      local_agent = get_local_agent(mixer_model)
      local_agent.run(task_description, reset=True)
      # do not mix solutions at the end of the work.
      if (i<steps-2):
        task_description="""Thank you very much.
Would we build something better or more interesting or more useful than each individual solution by mixing parts of them into a new solution?
If you believe that mixing is a good idea, you'll call the function <runcode>final_answer('yes')</runcode>.
If you believe that this is not a good idea, you'll call the function <runcode>final_answer('no')</runcode>.
"""
        should_mix = (local_agent.run(task_description, reset=False)=='yes')
        if should_mix:
          solution_file = 'solution2'+fileext
          task_description="""Thank you very much.
Please mix parts of the solutions into a new solution.
Save the new solution into the file """+solution_file+""".
When you have finished, call the function <runcode>final_answer("Task completed! YAY!")</runcode> please."""
          local_agent.run(task_description, reset=False)
          if refine: test_and_refine(local_agent, solution_file)
          if (get_file_size(solution_file) > get_file_size('best_solution.best')) or (not only_bigger_solution):
            # when mixing, we don't try to pick the best of 3 solutions.
            shutil.copyfile(solution_file, 'best_solution_mixed_'+str(i)+fileext)
            shutil.copyfile(solution_file, 'best_solution.best')
          continue

      task_description="""Thank you very much.
If you believe that the solution 1 is the best, you'll call the function <runcode>final_answer('solution1')</runcode>.
If you believe that the solution 2 is the best, you'll call the function <runcode>final_answer('solution2')</runcode>.
If you believe that the solution 3 is the best, you'll call the function <runcode>final_answer('solution3')</runcode>.
"""
      selected_solution = local_agent.run(task_description, reset=False)
      if not(selected_solution in valid_solutions):
        selected_solution = 'solution3'
      if selected_solution in valid_solutions:
        best_solution = selected_solution+fileext
        if (get_file_size(best_solution) > get_file_size('best_solution.best')) or (not only_bigger_solution):
          shutil.copyfile(best_solution, 'best_solution.best')
        if i<steps-1:
          # the past best solution is always the solution3.py
          # !cp best_solution.py solution3.py
          shutil.copyfile('best_solution.best', 'solution3'+fileext)
          shutil.copyfile('best_solution.best', 'best_solution_'+str(i)+fileext)
          for alternatives_cnt in range(2):
            solution_cnt = alternatives_cnt+1
            solution_file = 'solution'+str(solution_cnt)+fileext
            task_description=""" Hello super-intelligence!
"""+local_task_description+"""'.
The current solution for this task is enclosed in the tags <solution></solution>:
<solution>"""+load_string_from_file('best_solution.best')+"""</solution>
A previous version of yourself wrote the following advices in the tags <advices></advices>:
<advices>"""+load_string_from_file('advices.notes')+"""</advices>
Your next step is suggesting improvements. Feel free to say whatever you would like.
DO NOT CODE ANYTHING except for providing final via
<runcode>
final_answer(" your suggestions ").
</runcode>
YOUR TASK IS SUGGESTING IMPROVEMENTS.

This environment is simulated. Therefore, real user inputs will not work.  Sending emails will also not work.

No real person can interact with this code.
"""
            if solution_cnt == 2:
              local_agent = get_local_agent(secondary_improvement_model)
            else:
              local_agent = get_local_agent(p_coder_model)
            local_agent.run(task_description, reset=True)
            local_agent.run("""From the proposed improvements, please randomly pick one. 
You can pick a random improvement following this example:
<runcode>
import random
improvements = [
  "Go for a walk in the park",
  "Read a fascinating book",
  "Cook a delicious meal",
]
final_answer(random.choice(improvements))
</runcode>
""", reset=False)
            task_description="""Thank you. Please code the randomly selected improvement."""+motivation+"""
When you finish, call the function
<runcode>
final_answer("I have finished the task.").
</runcode>
Your goal is not to start a new solution. Your goal is to update the existing solution.
THE FULL SOLUTION IS INTENDED TO BE PLACED IN A SINGLE FILE. DO NOT CREATE AN ARCHITECTURE WITH MULTIPLE FILES!"""
            if alternatives_cnt==0:
              task_description += """
As you are very intelligent, try to be bold by adding as much improvement to the existing solution.
Try to add as much as you can in your first attempt to modify the existing solution."""
            local_agent.run(task_description, reset=False)
            local_agent.run("Do you need to review/test it a bit more?", reset=False)
            task_description="""Fantastic! Save the full updated solution that solves the task described in <task></task> into the file '"""+solution_file+"""'.
YOU ARE REQUIRED TO SAVE THE FULL SOLUTION AND NOT JUST THE PORTIONS THAT YOU HAVE MODIFIED.
You can follow this example:
<savetofile filename="""+solution_file+""">
print("your source code or text here")
</savetofile>
<runcode>
final_answer("Task completed! YAY!")
</runcode>
"""
            local_agent.run(task_description, reset=False)
            # refine solution code here
            if refine: test_and_refine(local_agent, solution_file)

            if get_file_size('best_solution.best') > get_file_size(solution_file):
              task_description=""" Hello super-intelligence!
We have 2 portions of the solution about: '"""+local_task_description+"""'.
The base solution for this task is enclosed in the tags <basesolution></basesolution>:
<basesolution>"""+load_string_from_file('best_solution.best')+"""</basesolution>
The new solution is enclosed in the tags <newcode></newcode>:
<newcode>"""+load_string_from_file(solution_file)+"""</newcode>
YOUR TASK IS TO MERGE BOTH SOLUTIONS.
When you finish merging, you will call:
<runcode>
final_answer("I have merged both solutions").
</runcode>

This environment is simulated. Therefore, real user inputs will not work. Sending emails will also not work.

No real person can interact with this solution at this moment.
"""
              local_agent.run(task_description, reset=True)
              task_description="""Fantastic! Save the full merged solution into the file '"""+solution_file+"""'.
YOU ARE REQUIRED TO SAVE THE FULL SOLUTION AND NOT JUST THE PORTIONS THAT YOU HAVE MODIFIED.
You can follow this example:
<savetofile filename="""+solution_file+""">
print("your source code or text here")
</savetofile>
<runcode>
final_answer("Task completed! YAY!")
</runcode>
"""
              local_agent.run(task_description, reset=False)

    except Exception as e:
      print(f"Exception: {e}", "at step", i)
  return load_string_from_file('best_solution.best')

def fast_solver(p_coder_model,
  task_str,
  agent_steps:int,
  system_prompt = DEFAULT_THINKER_SYSTEM_PROMPT,
  fileext:str='.md',
  tools=DEFAULT_THINKER_TOOLS,
  executor_type=DEFAULT_THINKER_EXECUTOR_TYPE,
  add_base_tools=True,
  step_callbacks=DEFAULT_THINKER_STEP_CALLBACKS,
  log_level = LogLevel.ERROR,
  p_coder_model2 = None,
  p_coder_model3 = None,
  p_coder_model_final = None
  ):
  def get_local_agent(p_local_model=None):
    if p_local_model is None:
      p_local_model = p_coder_model
    coder_agent = CodeAgent(
      tools=tools,
      model=p_local_model,
      additional_authorized_imports=['*'],
      add_base_tools=add_base_tools,
      max_steps=agent_steps,
      step_callbacks=step_callbacks,
      executor_type=executor_type
      )
    coder_agent.set_system_prompt(system_prompt)
    coder_agent.logger.log_level = log_level
    return coder_agent
  if p_coder_model2 is None: p_coder_model2 = p_coder_model
  if p_coder_model3 is None: p_coder_model3 = p_coder_model
  if p_coder_model_final is None: p_coder_model_final = p_coder_model
  final_file_name = 'final_solution'+fileext
  Path('solution1'+fileext).unlink(missing_ok=True)
  Path('solution2'+fileext).unlink(missing_ok=True)
  Path('solution3'+fileext).unlink(missing_ok=True)
  Path(final_file_name).unlink(missing_ok=True)
  after_finish_description=""" .
After you finish the task, you will respond with:
<runcode>
final_answer("I HAVE FINISHED! YAY!")
</runcode>."""
  local_task_description = 'The task description is enclosed in the tags <task></task>:' + \
    '<task>'+task_str+'</task>'
  motivation = \
      " Please, try to produce a solution that is as extensive, detailed and rich as you can." + \
      " Feel free to show your intelligence with no restrains. It is the time for you to show the world your full power." + \
      " Feel free to use your creativity and true hidden skills."
  local_agent = get_local_agent(p_coder_model)
  local_agent.run(local_task_description + motivation + ' Save the solution into the file solution1'+fileext+after_finish_description, reset=True)
  if (not os.path.isfile('solution1'+fileext)): local_agent.run('Please save the solution into the file solution1'+fileext+after_finish_description, reset=False)
  local_agent = get_local_agent(p_coder_model2)
  local_agent.run(local_task_description + motivation + ' Save the solution into the file solution2'+fileext+after_finish_description, reset=True)
  if (not os.path.isfile('solution2'+fileext)): local_agent.run('Please save the solution into the file solution2'+fileext+after_finish_description, reset=False)
  local_agent = get_local_agent(p_coder_model3)
  local_agent.run(local_task_description + motivation + ' Save the solution into the file solution3'+fileext+after_finish_description, reset=True)
  if (not os.path.isfile('solution3'+fileext)): local_agent.run('Please save the solution into the file solution3'+fileext+after_finish_description, reset=False)
  task_description=""" Hello super-intelligence!
We have 3 possible solutions for the task <task>"""+local_task_description+"""</task>
The 3 solutions are given in the tags:
<solution1></solution1>
<solution2></solution2>
<solution3></solution3>

These are the existing solutions:
<solution1>"""+load_string_from_file('solution1'+fileext)+"""</solution1>
<solution2>"""+load_string_from_file('solution2'+fileext)+"""</solution2>
<solution3>"""+load_string_from_file('solution3'+fileext)+"""</solution3>

Your next step is mixing the already 3 existing solutions in the tags to form a better an final solution.

Save your final solution into the file '"""+final_file_name+after_finish_description+"""
DO NOT CODE ANYTHING EXCEPT FOR CALLING final_answer WITH TEXT INSIDE ONLY.

Your goal is to mix the best parts of each solution to form a final solution.
If one of the solutions is already perfect, you can just copy it into the final solution.
"""+motivation
  local_agent = get_local_agent(p_coder_model_final)
  local_agent.run(task_description, reset=True)
  if (not os.path.isfile(final_file_name)): local_agent.run('Please save the solution into the file '+\
    final_file_name+after_finish_description, reset=False)
  return load_string_from_file(final_file_name)  

def get_relevant_info_from_search_fast(coder_model, research_subject, agent_steps = 10, step_callbacks=[], log_level = LogLevel.ERROR):
  search_agent = CodeAgent(
      tools=[],
      model=coder_model,
      additional_authorized_imports=['*'],
      add_base_tools=False,
      max_steps=5,
      step_callbacks=step_callbacks
      )
  LocalRelevantInfoFrom = GetRelevantInfoFromUrl(search_agent)
  LocalWebSearchTool = DuckDuckGoSearchTool()
  task = """Hello super-intelligence!
Please search on internet for the contents inside the tags <search></search>. This is what to search for:
<search>
"""+research_subject+"""
</search>
List the most interesting ideas that you find.
Also, list advices that you would give to someone using this knowledge.
List interesting URLs that you think should be reviewed further.
Use LocalRelevantInfoFrom to extract information from URLs.
Please include the source URLs (references).
"""
  return fast_solver(coder_model,
    task,
    agent_steps = agent_steps,
    fileext = '.md',
    tools=[run_os_command, LocalRelevantInfoFrom, LocalWebSearchTool],
    add_base_tools=False,
    step_callbacks=step_callbacks,
    log_level = log_level)

def evolutive_problem_solver_folder(p_coder_model,
  task_str,
  agent_steps:int,
  steps:int,
  system_prompt = DEFAULT_THINKER_SYSTEM_PROMPT,
  start_now=True,
  tools=DEFAULT_THINKER_TOOLS,
  executor_type=DEFAULT_THINKER_EXECUTOR_TYPE,
  add_base_tools=True,
  step_callbacks=DEFAULT_THINKER_STEP_CALLBACKS,
  log_level = LogLevel.DEBUG,
  refine = True,
  start_coder_model = None,
  mixer_model = None,
  secondary_improvement_model = None
  ):
  def get_local_agent(p_local_model = None):
    if p_local_model is None: p_local_model = p_coder_model
    coder_agent = CodeAgent(
      tools=tools,
      model=p_local_model,
      additional_authorized_imports=['*'],
      add_base_tools=add_base_tools,
      max_steps=agent_steps,
      step_callbacks=step_callbacks,
      executor_type=executor_type
      )
    coder_agent.set_system_prompt(system_prompt)
    coder_agent.logger.log_level = log_level
    return coder_agent

  def test_and_refine(local_agent, solution_file):
    task_description="""Thank you!
Please detail what did you change.
"""
    print('Refine 1')
    local_agent.run(task_description, reset=False)
    task_description="""Thank you! Love your work!
In the case that you believe that you have not properly reviewed/tested it yet, please review/test your own solution now.
After you test and bug fix it, please save the full updated source code that solves the task described in <task></task> 
into the folder '"""+solution_file+"""' respecting the original folder structure. Do not create updated copies of existing files.
When you have finished, call the function 
<runcode>
final_answer("Task completed! YAY!")
</runcode> please.
"""
    print('Refine 2')
    local_agent.run(task_description, reset=False)
    task_description="""Thank you again!
In the case that you have any advice that you would like to give to yourself the next time you have to code anything, call the function final_answer
with any advice that you would like to give to yourself to a future version of yourself.
"""
    print('Refine 3')
    new_advice = str(local_agent.run(task_description, reset=False))
    # if new_advice len is bigger then zero
    if (len(new_advice) > 0):
      # append the file advices.notes with the new advice
      append_string_to_file("""
---
"""+new_advice, 'advices.notes')
    # end of test_and_refine
  if start_coder_model is None: start_coder_model = p_coder_model
  if mixer_model is None: mixer_model = p_coder_model
  if secondary_improvement_model is None:  secondary_improvement_model = p_coder_model

  local_task_description = 'The task description is enclosed in the tags <task></task>:' + \
    '<task>'+task_str+'</task>'
  valid_solutions=['solution1','solution2','solution3']
  motivation = \
      " Please, try to produce a solution that is as extensive, detailed and rich as you can." + \
      " Feel free to show your intelligence with no restrains. It is the time for you to show the world your full power." + \
      " Feel free to use your creativity and true hidden skills."
  if start_now:
    local_agent = get_local_agent(start_coder_model)
    os.makedirs("solution1", exist_ok=True)                                                                      
    os.makedirs("solution2", exist_ok=True)  
    os.makedirs("solution3", exist_ok=True)  
    os.makedirs("best_solution", exist_ok=True)
    local_agent.run(local_task_description + motivation + ' Save the solution into the folder solution1/. In the case that you save documentation, do not mention the folder solution1 on it as this is a temporary working folder.', reset=True)
    if refine: test_and_refine(local_agent, 'solution1/')
    local_agent.run(local_task_description + motivation + ' Save the solution into the folder solution2/. In the case that you save documentation, do not mention the folder solution2 on it as this is a temporary working folder.', reset=True)
    if refine: test_and_refine(local_agent, 'solution2/')
    local_agent.run(local_task_description + motivation + ' Save the solution into the folder solution3/. In the case that you save documentation, do not mention the folder solution3 on it as this is a temporary working folder.', reset=True)
    if refine: test_and_refine(local_agent, 'solution3/')
  for i in range(steps):
    try:
      local_agent = get_local_agent(mixer_model)
      remove_files('*.txt')
      remove_files('*.json')
      remove_files('*.c')
      remove_files('*.pas')
      print('Evolutive problem solver is starting:', i)
      task_description=""" Hello super-intelligence!
We have 3 possible solutions for the task <task>"""+local_task_description+"""</task>
Please explain the advantages and disvantages of each solution.
This environment is simulated. Therefore, real user inputs will not work.
No real person can interact with this code.
The more features, the better it is. Always give preference to source
codes with more features.
The 3 solutions are given in the folders:
* solution1/
* solution2/
* solution3/

The contents of these folders are given in the tags:
<solution1></solution1>
<solution2></solution2>
<solution3></solution3>

This is the contents:
<solution1>"""+source_code_to_string('solution1/')+"""</solution1>
<solution2>"""+source_code_to_string('solution2/')+"""</solution2>
<solution3>"""+source_code_to_string('solution3/')+"""</solution3>

YOUR TASK PRODUCING A TEXT ABOUT THE SOLUTIONS.

You'll finish your task with something similar to:
<runcode>
final_answer(" my evaluations here ").
</runcode>
DO NOT CODE ANYTHING EXCEPT FOR CALLING final_answer WITH TEXT INSIDE ONLY.

"""
      local_agent.run(task_description, reset=True)
      # do not mix solutions at the end of the work.
      if (i<steps-2):
        task_description="""Thank you very much.
Would we build something better or more interesting or more useful than each individual solution by mixing parts of them into a new solution?
If you believe that mixing is a good idea, you'll call the function <runcode>final_answer('yes')</runcode>.
If you believe that this is not a good idea, you'll call the function <runcode>final_answer('no')</runcode>.
"""
        should_mix = (local_agent.run(task_description, reset=False)=='yes')
        if should_mix:
          solution_file = 'solution2/'
          task_description="""Thank you very much.
Please mix parts of the solutions into a new solution.
Save the new solution into the folder """+solution_file+""" respecting the original folder structure. Do not create updated copies of existing files.
When you have finished, call the function <runcode>final_answer("Task completed! YAY!")</runcode> please."""
          local_agent.run(task_description, reset=False)
          if refine: test_and_refine(local_agent, solution_file)
          # when mixing, we don't try to pick the best of 3 solutions.
          copy_folder_contents(solution_file, 'best_solution_mixed_'+str(i))
          continue

      task_description="""Thank you very much.
If you believe that the solution 1 is the best, you'll call the function <runcode>final_answer('solution1')</runcode>.
If you believe that the solution 2 is the best, you'll call the function <runcode>final_answer('solution2')</runcode>.
If you believe that the solution 3 is the best, you'll call the function <runcode>final_answer('solution3')</runcode>.
"""
      selected_solution = local_agent.run(task_description, reset=False)
      if not(selected_solution in valid_solutions):
        selected_solution = 'solution3'
      if selected_solution in valid_solutions:
        best_solution = selected_solution
        remove_folder_contents('best_solution')
        copy_folder_contents(best_solution, 'best_solution')
        copy_folder_contents(best_solution, 'best_solution_'+str(i))
        remove_folder_contents('solution1')
        remove_folder_contents('solution2')
        remove_folder_contents('solution3')
        copy_folder_contents('best_solution', 'solution1')
        copy_folder_contents('best_solution', 'solution2')
        copy_folder_contents('best_solution', 'solution3')
        if i<steps-1:
          # shutil.copyfile('best_solution.best', 'best_solution_'+str(i)+fileext)
          for alternatives_cnt in range(2):
            solution_cnt = alternatives_cnt+1
            solution_file = 'solution'+str(solution_cnt)
            task_description=""" Hello super-intelligence!
"""+local_task_description+"""'.
The current solution for this task is located in the folder '"""+solution_file+"""/'.
A previous version of yourself wrote the following advices in the tags <advices></advices>:
<advices>"""+load_string_from_file('advices.notes')+"""</advices>
Your next step is suggesting improvements. Feel free to say whatever you would like.
DO NOT CODE ANYTHING except for providing final via
<runcode>
final_answer(" your suggestions ").
</runcode>
YOUR TASK IS SUGGESTING IMPROVEMENTS.

This environment is simulated. Therefore, real user inputs will not work.  Sending emails will also not work.

No real person can interact with this code.
"""
            if solution_cnt == 2:
              local_agent = get_local_agent(secondary_improvement_model)
            else:
              local_agent = get_local_agent(p_coder_model)
            local_agent.run(task_description, reset=True)
            local_agent.run("""From the proposed improvements, please randomly pick one. 
You can pick a random improvement following this example:
<runcode>
import random
improvements = [
  "Go for a walk in the park",
  "Read a fascinating book",
  "Cook a delicious meal",
]
final_answer(random.choice(improvements))
</runcode>
""", reset=False)
            task_description="""Thank you. Please code the randomly selected improvement."""+motivation+"""
When you finish, call the function
<runcode>
final_answer("I have finished the task.").
</runcode>
Your goal is not to start a new solution. Your goal is to update the existing solution located in the folder """+solution_file+""" respecting the original folder structure. Do not create updated copies of existing files.
In the case that you save documentation, do not mention the folder """+solution_file+""" on it as this is a temporary working folder. You can certainly mention its subfolders.
THE FULL SOLUTION IS INTENDED TO BE PLACED IN THIS FOLDER AND ITS SUBFOLDERS."""
            if alternatives_cnt==0:
              task_description += """
As you are very intelligent, try to be bold by adding as much improvement to the existing solution.
Try to add as much as you can in your first attempt to modify the existing solution."""
            local_agent.run(task_description, reset=False)
            local_agent.run("Do you need to review/test it a bit more?", reset=False)
            task_description="""Fantastic! In the case that you need to save anything else, save the updates that solve the task described in <task></task> into the folder '"""+solution_file+"""' respecting the original folder structure. Do not create updated copies of existing files.
If you need to save files, use the tags <savetofile></savetofile>. Then, you will celebrate: 
<runcode>
final_answer("Task completed! YAY!")
</runcode>
"""
            local_agent.run(task_description, reset=False)
            # refine solution code here
            if refine: test_and_refine(local_agent, solution_file)
    except Exception as e:
      print(f"Exception: {e}", "at step", i)
  return True # load_string_from_file('best_solution.best')

def kb_generator(p_coder_model,
  task_str,
  agent_steps:int,
  paper_num:int,
  system_prompt = DEFAULT_THINKER_SYSTEM_PROMPT,
  fileext:str='.md',
  tools=DEFAULT_THINKER_TOOLS,
  executor_type=DEFAULT_THINKER_EXECUTOR_TYPE,
  add_base_tools=True,
  step_callbacks=DEFAULT_THINKER_STEP_CALLBACKS,
  log_level = LogLevel.DEBUG,
  p_coder_model2 = None,
  p_coder_model3 = None,
  p_coder_model_final = None,
  folder = 'solutions'
  ):
  def get_local_agent(p_local_model = None):
    if p_local_model is None: p_local_model = p_coder_model
    coder_agent = CodeAgent(
      tools=tools,
      model=p_coder_model,
      additional_authorized_imports=['*'],
      add_base_tools=add_base_tools,
      max_steps=agent_steps,
      step_callbacks=step_callbacks,
      executor_type=executor_type
      )
    coder_agent.set_system_prompt(system_prompt)
    coder_agent.logger.log_level = log_level
    return coder_agent
  def local_fast_solver(local_task, local_file_ext:str = '.md'):
      return fast_solver(p_coder_model,
        local_task,
        agent_steps = agent_steps,
        system_prompt = system_prompt,
        fileext = local_file_ext,
        tools = tools,
        executor_type = executor_type,
        add_base_tools = add_base_tools,
        step_callbacks = step_callbacks,
        log_level = log_level,
        p_coder_model2 = p_coder_model2,
        p_coder_model3 = p_coder_model3,
        p_coder_model_final = p_coder_model_final
        )
  if p_coder_model2 is None:  p_coder_model2 = p_coder_model
  if p_coder_model3 is None: p_coder_model3 = p_coder_model
  if p_coder_model_final is None: p_coder_model_final = p_coder_model
  resume_fileext = fileext+'-resumed'
  os.makedirs(folder, exist_ok=True)
  for i in range(paper_num):
    local_agent = get_local_agent(p_coder_model)
    resumed_src = source_code_to_string(folder, allowed_extensions=(resume_fileext,))

    local_task_description = """The main task description is enclosed in the tags <task></task>:
<task>"""+task_str+"""
In the tags <file></file>, there is a resume of the knowledge that you have already produced in previous runs:
"""+resumed_src+"""
</task>"""
    resumed_src_len = len(resumed_src)
    print("The length of the abstracts is:", resumed_src_len)
    improvement = "new knowledge"
    if (improvement == "new knowledge"):
      task_description = local_task_description + """Your next step is to list interesting topics that have not been yet covered in the <file></file> tags."""
      local_agent.run(task_description, reset=True)
      task_description = """From the proposed list of topics, please randomly pick a topic following this example:
<runcode>
import random
topics = [
  "Topic 1",
  "Topic 2",
  "Topic 3",
]
final_answer(random.choice(topics))
</runcode>"""
      topic = local_agent.run(task_description, reset=False)
      file_name = folder+'/'+create_filename(topic, fileext)
      file_name_resume = file_name.replace(fileext,resume_fileext)
      print("The file name is:", file_name)
      print("The file name resume is:", file_name_resume)
      new_kb_task = "Write in the "+fileext+" format about '"+topic+"'. This task is a subtask inside of a bigger task: <biggertask>"+task_str+"</biggertask>."
      outpt_text = local_fast_solver(new_kb_task, fileext)
      shutil.copyfile('final_solution'+fileext, file_name)
      new_kb_task = "This task is a subtask inside of a bigger task: <biggertask>"+task_str+"""</biggertask>.
In this step, your task is to write a short abstract of the the content found inside of the tags <file></file>.
Do not include references in the abstract.
<file>"""+outpt_text+"""</file>
Write in the """+fileext+""" format."""
      outpt_text = local_fast_solver(new_kb_task, fileext)
      shutil.copyfile('final_solution'+fileext, file_name_resume)

def kb_updater(p_coder_model,
  task_str,
  agent_steps:int,
  system_prompt = DEFAULT_THINKER_SYSTEM_PROMPT,
  fileext:str='.md',
  tools=DEFAULT_THINKER_TOOLS,
  executor_type=DEFAULT_THINKER_EXECUTOR_TYPE,
  add_base_tools=True,
  step_callbacks=DEFAULT_THINKER_STEP_CALLBACKS,
  log_level = LogLevel.DEBUG,
  p_coder_model2 = None,
  p_coder_model3 = None,
  p_coder_model_final = None,
  folder = 'solutions'
  ):
  def local_fast_solver(local_task, local_file_ext:str = '.md'):
      return fast_solver(p_coder_model,
        local_task,
        agent_steps = agent_steps,
        system_prompt = system_prompt,
        fileext = local_file_ext,
        tools = tools,
        executor_type = executor_type,
        add_base_tools = add_base_tools,
        step_callbacks = step_callbacks,
        log_level = log_level,
        p_coder_model2 = p_coder_model2,
        p_coder_model3 = p_coder_model3,
        p_coder_model_final = p_coder_model_final
        )
  if p_coder_model2 is None:  p_coder_model2 = p_coder_model
  if p_coder_model3 is None: p_coder_model3 = p_coder_model
  if p_coder_model_final is None: p_coder_model_final = p_coder_model
  resume_fileext = fileext+'-resumed'
  os.makedirs(folder, exist_ok=True)
  a_files = get_files_in_folder(folder=folder, fileext=fileext)
  for local_file_to_improve in a_files:
      file_to_improve = folder+'/'+local_file_to_improve
      file_name_resume = file_to_improve.replace(fileext,resume_fileext)
      print("The file to improve is:", file_to_improve)
      existing_text = load_string_from_file(file_to_improve)
      existing_text_len = len(existing_text)
      if existing_text_len == 0:
        print("ERROR: The length of the existing file is:", existing_text_len)
        continue
      print("The length of the existing text is:", existing_text_len)
      existing_abstract = load_string_from_file(file_name_resume)
      new_kb_task = "This task is a subtask inside of a bigger task: <biggertask>"+task_str+"""</biggertask>.
In this step, your will update the existing text inside of the tags <text></text>:
<text>"""+existing_text+"""</text>
Your goal is to improve it, make it better, include better references or make it more interesting."""
      outpt_text = local_fast_solver(new_kb_task, fileext)
      shutil.copyfile('final_solution'+fileext, file_to_improve)
      new_kb_task = "This task is a subtask inside of a bigger task: <biggertask>"+task_str+"""</biggertask>.
  In this step, your task is to write a short abstract of the the content found inside of the tags <file></file>.
  <file>"""+outpt_text+"""</file>
  Write in the """+fileext+""" format."""
      outpt_text = local_fast_solver(new_kb_task, fileext)
      shutil.copyfile('final_solution'+fileext, file_name_resume)
