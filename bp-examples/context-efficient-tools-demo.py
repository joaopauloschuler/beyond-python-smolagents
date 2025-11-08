#!/usr/bin/env python3
"""
Demo: Context-Efficient Tools for AI Agents

This example demonstrates the new context-efficient tools added to Beyond Python Smolagents.
These tools help agents work with codebases efficiently without burning their context window.

The new tools include:
1. list_directory_tree - Get project structure overview
2. search_in_files - Find code patterns without loading all files
3. extract_function_signatures - Understand code structure efficiently
4. get_file_info - Check file metadata before loading
5. count_lines_of_code - Get project metrics
6. list_directory - List files with pattern matching
7. compare_files - See differences between files
8. read_file_range - Read only parts of large files
9. mkdir, delete_file, delete_directory - File system operations

These tools are designed to help agents:
- Understand project structure quickly
- Find relevant code without loading everything
- Work with large files efficiently
- Make informed decisions about what to load into context
"""

import os
import sys


# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents.bp_tools import (
    count_lines_of_code,
    delete_directory,
    extract_function_signatures,
    get_file_info,
    list_directory,
    list_directory_tree,
    mkdir,
    search_in_files,
)


def demo_directory_tree():
    """Demo 1: Understanding project structure without loading all files"""
    print("\n" + "=" * 80)
    print("DEMO 1: Project Structure Overview")
    print("=" * 80)
    print("\nInstead of loading all files, first get an overview:")
    print("\nlist_directory_tree('src/smolagents', max_depth=2, show_files=True)")

    tree = list_directory_tree('src/smolagents', max_depth=2, show_files=True)
    print(tree[:800] + "...")

    print("\n💡 Use Case: Before diving into code, understand the project structure")
    print("   This saves context by showing you what exists without loading content.")

def demo_search_in_files():
    """Demo 2: Finding relevant code without loading all files"""
    print("\n" + "=" * 80)
    print("DEMO 2: Efficient Code Search")
    print("=" * 80)
    print("\nInstead of loading all Python files, search for specific patterns:")
    print("\nsearch_in_files('src/smolagents', 'class.*Tool', file_extensions=('.py',), max_results=5)")

    results = search_in_files(
        'src/smolagents',
        'class.*Tool',
        file_extensions=('.py',),
        case_sensitive=False,
        max_results=5
    )
    print(results)

    print("\n💡 Use Case: Find where specific functions/classes are defined")
    print("   Only load the relevant files after finding them.")

def demo_function_signatures():
    """Demo 3: Understanding code structure efficiently"""
    print("\n" + "=" * 80)
    print("DEMO 3: Extract Function Signatures")
    print("=" * 80)
    print("\nGet an overview of functions without loading full implementations:")
    print("\nextract_function_signatures('src/smolagents/bp_tools.py', 'python')")

    signatures = extract_function_signatures('src/smolagents/bp_tools.py', 'python')
    lines = signatures.split('\n')
    print('\n'.join(lines[:20]) + '\n...')

    print("\n💡 Use Case: Understand what functions exist before deciding which to examine")
    print("   Get the API surface without implementation details.")

def demo_file_info():
    """Demo 4: Check file metadata before loading"""
    print("\n" + "=" * 80)
    print("DEMO 4: File Metadata Inspection")
    print("=" * 80)
    print("\nCheck file properties before deciding to load it:")
    print("\nget_file_info('README.md')")

    info = get_file_info('README.md')
    print(f"File exists: {info['exists']}")
    print(f"Size: {info['size_bytes']:,} bytes ({info['size_bytes']/1024:.1f} KB)")
    print(f"Is readable: {info['readable']}")

    print("\n💡 Use Case: Avoid loading huge files into context accidentally")
    print("   Check size first, maybe use read_file_range for large files.")

def demo_code_metrics():
    """Demo 5: Understanding project size"""
    print("\n" + "=" * 80)
    print("DEMO 5: Project Code Metrics")
    print("=" * 80)
    print("\nGet an overview of codebase size:")
    print("\ncount_lines_of_code('src/smolagents', ('.py',))")

    counts = count_lines_of_code('src/smolagents', file_extensions=('.py',))
    for ext, count in sorted(counts.items()):
        if ext == '_total':
            print(f"\nTotal: {count:,} lines")
        else:
            print(f"  {ext}: {count:,} lines")

    print("\n💡 Use Case: Understand project scale before diving in")
    print("   Helps plan how to approach the codebase.")

def demo_list_directory():
    """Demo 6: Filtered file listing"""
    print("\n" + "=" * 80)
    print("DEMO 6: Pattern-Based File Listing")
    print("=" * 80)
    print("\nFind files matching patterns:")
    print("\nlist_directory('src/smolagents', pattern='*.py', files_only=True)")

    files = list_directory('src/smolagents', pattern='*tools*.py', files_only=True)
    print(f"Found {len(files)} files with 'tools' in the name:")
    for f in files[:5]:
        print(f"  {f}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")

    print("\n💡 Use Case: Find specific types of files without recursive loading")

def demo_workflow():
    """Demo 7: Efficient agent workflow"""
    print("\n" + "=" * 80)
    print("DEMO 7: Recommended Agent Workflow")
    print("=" * 80)

    print("\n📋 Efficient Workflow for Coding Tasks:")
    print("""
1. 📁 Get project structure (list_directory_tree)
   └─> Understand organization without loading files
   
2. 🔍 Search for relevant code (search_in_files)
   └─> Find specific functions/patterns efficiently
   
3. 📊 Check file metadata (get_file_info)
   └─> Avoid loading huge files accidentally
   
4. 🎯 Extract signatures (extract_function_signatures)
   └─> Understand APIs without implementations
   
5. 📖 Load only necessary files (load_string_from_file)
   └─> Now you know exactly what to load
   
6. ⚙️  Make changes (replace_on_file, etc.)
   └─> Work with confidence
   
7. 🔄 Compare results (compare_files)
   └─> Verify your changes

This workflow uses ~10x less context than "load everything first"!
    """)

def demo_file_operations():
    """Demo 8: File system operations"""
    print("\n" + "=" * 80)
    print("DEMO 8: File System Operations")
    print("=" * 80)
    print("\nCreate and manage directories:")

    # Create a test directory structure
    test_dir = '/tmp/bp_demo_test'

    # Clean up if exists
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)

    print(f"\nmkdir('{test_dir}/subdir/deep', parents=True)")
    mkdir(f'{test_dir}/subdir/deep', parents=True)
    print("✓ Directory created")

    # Create a test file
    test_file = f'{test_dir}/test.txt'
    with open(test_file, 'w') as f:
        f.write("Test content")

    print(f"\nFile created: {test_file}")
    print(f"list_directory('{test_dir}', recursive=True)")
    files = list_directory(test_dir, recursive=True)
    for f in files:
        print(f"  {f}")

    # Clean up
    print(f"\ndelete_directory('{test_dir}', recursive=True)")
    delete_directory(test_dir, recursive=True)
    print("✓ Cleaned up")

    print("\n💡 Use Case: Manage project structure dynamically")

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("CONTEXT-EFFICIENT TOOLS DEMO")
    print("Beyond Python Smolagents - AI Agents That Code Efficiently")
    print("=" * 80)

    # Change to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')
    os.chdir(project_root)
    print(f"\nWorking directory: {os.getcwd()}")

    try:
        demo_directory_tree()
        demo_search_in_files()
        demo_function_signatures()
        demo_file_info()
        demo_code_metrics()
        demo_list_directory()
        demo_file_operations()
        demo_workflow()

        print("\n" + "=" * 80)
        print("✅ DEMO COMPLETED!")
        print("=" * 80)
        print("\nThese tools help AI agents:")
        print("  • Understand codebases without loading everything")
        print("  • Find relevant code efficiently")
        print("  • Make informed decisions about what to load")
        print("  • Work with large projects within context limits")
        print("\n" + "=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
