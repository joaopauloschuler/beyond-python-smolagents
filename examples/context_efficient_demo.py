#!/usr/bin/env python3
"""
Demonstration of context-efficient evolutive_problem_solver_folder

This script shows the before/after comparison of context usage.
"""

import os
import sys
import tempfile

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents.bp_tools import (
    source_code_to_string,
    list_directory_tree,
    count_lines_of_code,
    save_string_to_file
)


def create_sample_solution(solution_dir, solution_num):
    """Create a sample solution directory structure"""
    os.makedirs(solution_dir, exist_ok=True)
    
    # Create a src subdirectory
    src_dir = os.path.join(solution_dir, 'src')
    os.makedirs(src_dir, exist_ok=True)
    
    # Create multiple Python files
    for i in range(10):
        file_path = os.path.join(src_dir, f'module_{i}.py')
        content = f'''"""
Module {i} for Solution {solution_num}
"""

class TaskManager{i}:
    """A task management class"""
    
    def __init__(self):
        self.tasks = []
        self.solution_id = {solution_num}
        self.module_id = {i}
    
    def add_task(self, task):
        """Add a task to the manager"""
        self.tasks.append(task)
        return len(self.tasks)
    
    def remove_task(self, task_id):
        """Remove a task by ID"""
        if 0 <= task_id < len(self.tasks):
            return self.tasks.pop(task_id)
        return None
    
    def get_task(self, task_id):
        """Get a task by ID"""
        if 0 <= task_id < len(self.tasks):
            return self.tasks[task_id]
        return None
    
    def list_tasks(self):
        """List all tasks"""
        return self.tasks.copy()
    
    def complete_task(self, task_id):
        """Mark a task as complete"""
        task = self.get_task(task_id)
        if task:
            task['completed'] = True
        return task

def process_tasks(manager, task_list):
    """Process a list of tasks"""
    for task in task_list:
        manager.add_task(task)
    return manager.list_tasks()
'''
        save_string_to_file(content, file_path)
    
    # Create a README
    readme_path = os.path.join(solution_dir, 'README.md')
    readme_content = f'''# Solution {solution_num}

This is solution {solution_num} for the task management system.

## Features
- Task creation and management
- Multiple task managers (modules 0-9)
- Task completion tracking
- Efficient task processing

## Usage
```python
from src.module_0 import TaskManager0

manager = TaskManager0()
manager.add_task({{"title": "Example task"}})
```
'''
    save_string_to_file(readme_content, readme_path)


def main():
    """Run the demonstration"""
    print("=" * 70)
    print("Context-Efficient evolutive_problem_solver_folder Demonstration")
    print("=" * 70)
    print()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 solution directories
        print("Creating 3 sample solution directories...")
        for i in range(1, 4):
            solution_dir = os.path.join(tmpdir, f'solution{i}')
            create_sample_solution(solution_dir, i)
        
        print("✓ Created solution1/, solution2/, solution3/")
        print()
        
        # OLD APPROACH: Load full content
        print("-" * 70)
        print("OLD APPROACH: Load full content with source_code_to_string()")
        print("-" * 70)
        
        old_content = ""
        for i in range(1, 4):
            solution_dir = os.path.join(tmpdir, f'solution{i}')
            content = source_code_to_string(solution_dir)
            old_content += f"<solution{i}>{content}</solution{i}>\n"
        
        old_size = len(old_content)
        old_lines = old_content.count('\n')
        
        print(f"Total characters loaded: {old_size:,}")
        print(f"Total lines loaded: {old_lines:,}")
        print(f"Sample (first 200 chars):")
        print(old_content[:200])
        print("...")
        print()
        
        # NEW APPROACH: Structure + metrics
        print("-" * 70)
        print("NEW APPROACH: Structure + metrics (context-efficient)")
        print("-" * 70)
        
        new_content = ""
        for i in range(1, 4):
            solution_dir = os.path.join(tmpdir, f'solution{i}')
            tree = list_directory_tree(solution_dir, max_depth=3, show_files=True)
            metrics = count_lines_of_code(solution_dir)
            
            new_content += f"""
<solution{i}_structure>
{tree}
Lines of code metrics: {metrics}
</solution{i}_structure>

"""
        
        new_size = len(new_content)
        new_lines = new_content.count('\n')
        
        print(f"Total characters loaded: {new_size:,}")
        print(f"Total lines loaded: {new_lines:,}")
        print(f"Full content:")
        print(new_content)
        print()
        
        # Calculate savings
        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)
        
        savings_chars = old_size - new_size
        savings_percent = (savings_chars / old_size * 100) if old_size > 0 else 0
        reduction_factor = old_size / new_size if new_size > 0 else 0
        
        print(f"Characters saved: {savings_chars:,} ({savings_percent:.1f}%)")
        print(f"Lines saved: {old_lines - new_lines:,}")
        print(f"Reduction factor: {reduction_factor:.1f}x")
        print()
        
        print("=" * 70)
        print("KEY BENEFITS")
        print("=" * 70)
        print("✓ Agent can still see project structure and organization")
        print("✓ Agent knows the scale (lines of code by type)")
        print("✓ Agent has access to context-efficient tools:")
        print("  - list_directory_tree() - Explore structure")
        print("  - search_in_files() - Find specific patterns")
        print("  - extract_function_signatures() - See APIs")
        print("  - get_file_info() - Check file sizes")
        print("  - load_string_from_file() - Load when needed")
        print("✓ Dramatically reduced context consumption")
        print("✓ Can handle much larger projects")
        print()


if __name__ == '__main__':
    main()
