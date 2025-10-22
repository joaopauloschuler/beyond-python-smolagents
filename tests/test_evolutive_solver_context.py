#!/usr/bin/env python3
"""
Integration test for context-efficient evolutive_problem_solver_folder
"""

import os
import sys
import tempfile
import shutil

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents.bp_tools import (
    list_directory_tree,
    count_lines_of_code,
    save_string_to_file
)


def test_context_efficiency_improvement():
    """
    Test that demonstrates the context efficiency improvement.
    
    This test creates sample solution folders and measures:
    1. The size of full content loading (old approach)
    2. The size of structure + metrics (new approach)
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create three solution folders with sample files
        for sol_num in range(1, 4):
            sol_dir = os.path.join(tmpdir, f'solution{sol_num}')
            os.makedirs(sol_dir, exist_ok=True)
            
            # Create some sample Python files
            for file_num in range(5):
                file_path = os.path.join(sol_dir, f'module{file_num}.py')
                content = f"""# Solution {sol_num} - Module {file_num}
def function_{sol_num}_{file_num}():
    '''A sample function'''
    print("Hello from solution {sol_num}, module {file_num}")
    return {sol_num} * {file_num}

class Class{sol_num}_{file_num}:
    def __init__(self):
        self.value = {sol_num} + {file_num}
    
    def method1(self):
        return self.value * 2
    
    def method2(self):
        return self.value ** 2
"""
                save_string_to_file(content, file_path)
        
        # Measure OLD approach: Load full content
        from smolagents.bp_tools import source_code_to_string
        
        old_approach_content = ""
        for sol_num in range(1, 4):
            sol_dir = os.path.join(tmpdir, f'solution{sol_num}')
            old_approach_content += source_code_to_string(sol_dir)
        
        old_size = len(old_approach_content)
        
        # Measure NEW approach: Structure + metrics
        new_approach_content = ""
        for sol_num in range(1, 4):
            sol_dir = os.path.join(tmpdir, f'solution{sol_num}')
            tree = list_directory_tree(sol_dir, max_depth=3, show_files=True)
            metrics = count_lines_of_code(sol_dir)
            new_approach_content += f"Structure:\n{tree}\nMetrics: {metrics}\n\n"
        
        new_size = len(new_approach_content)
        
        # Calculate savings
        savings_percent = ((old_size - new_size) / old_size * 100) if old_size > 0 else 0
        
        print(f"\n=== Context Efficiency Test Results ===")
        print(f"Old approach (full content): {old_size:,} characters")
        print(f"New approach (structure + metrics): {new_size:,} characters")
        print(f"Context savings: {savings_percent:.1f}%")
        print(f"Reduction factor: {old_size / new_size:.1f}x")
        
        # Assertions
        assert new_size < old_size, "New approach should use less context"
        assert savings_percent > 80, f"Should save at least 80% of context, but saved {savings_percent:.1f}%"
        
        print("\n✅ Test passed! Context efficiency significantly improved.")


def test_context_efficient_information_preserved():
    """
    Test that the new approach still provides useful information for decision-making.
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a solution folder
        sol_dir = os.path.join(tmpdir, 'solution1')
        os.makedirs(sol_dir, exist_ok=True)
        
        # Create files
        save_string_to_file("def test():\n    pass\n", os.path.join(sol_dir, 'test.py'))
        save_string_to_file("print('hello')\n", os.path.join(sol_dir, 'main.py'))
        
        # Get structure and metrics
        tree = list_directory_tree(sol_dir, max_depth=3, show_files=True)
        metrics = count_lines_of_code(sol_dir)
        
        # Verify useful information is present
        assert 'test.py' in tree, "Structure should show test.py"
        assert 'main.py' in tree, "Structure should show main.py"
        assert '.py' in metrics, "Metrics should include Python files"
        assert metrics['_total'] > 0, "Total line count should be positive"
        
        print("\n✅ Information preservation test passed!")


if __name__ == '__main__':
    test_context_efficiency_improvement()
    test_context_efficient_information_preserved()
