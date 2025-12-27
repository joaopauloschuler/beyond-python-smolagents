#!/usr/bin/env python3
"""
Unit tests for context-efficient tools in bp_tools.py
"""

import os
import sys

import pytest


# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents.bp_tools import (
    compare_files,
    compare_folders,
    count_lines_of_code,
    delete_directory,
    delete_file,
    delete_lines_from_file,
    extract_function_signatures,
    get_file_info,
    list_directory,
    list_directory_tree,
    mkdir,
    read_file_range,
    read_first_n_lines,
    read_last_n_lines,
    search_in_files,
)


class TestListDirectoryTree:
    def test_basic_tree(self, tmp_path):
        """Test basic directory tree generation"""
        # Create test structure
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir1" / "file1.txt").write_text("content")
        (tmp_path / "dir2").mkdir()

        result = list_directory_tree(str(tmp_path), max_depth=2, show_files=True)

        assert "dir1/" in result
        assert "dir2/" in result
        assert "file1.txt" in result

    def test_max_depth(self, tmp_path):
        """Test max depth limiting"""
        # Create nested structure
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)

        result = list_directory_tree(str(tmp_path), max_depth=1)
        assert "a/" in result
        assert "b/" not in result or result.count("/b/") == 0

    def test_files_only_false(self, tmp_path):
        """Test showing only directories"""
        (tmp_path / "dir1").mkdir()
        (tmp_path / "file1.txt").write_text("content")

        result = list_directory_tree(str(tmp_path), show_files=False)
        assert "dir1/" in result
        assert "file1.txt" not in result

    def test_invalid_directory(self):
        """Test handling of invalid directory"""
        result = list_directory_tree("/nonexistent/path")
        assert "not a valid directory" in result

    def test_line_count_for_source_files(self, tmp_path):
        """Test that source code files show line counts"""
        # Create Python source file
        (tmp_path / "test.py").write_text("line1\nline2\nline3\n")

        result = list_directory_tree(str(tmp_path), max_depth=1, show_files=True)

        assert "test.py (3 lines)" in result
        assert "Total source code lines: 3" in result

    def test_line_count_for_multiple_extensions(self, tmp_path):
        """Test that various source code extensions are counted"""
        # Create files with different source code extensions
        (tmp_path / "script.py").write_text("line1\nline2\n")
        (tmp_path / "styles.css").write_text("line1\nline2\nline3\n")
        (tmp_path / "config.json").write_text("line1\n")
        (tmp_path / "readme.md").write_text("line1\nline2\nline3\nline4\n")

        result = list_directory_tree(str(tmp_path), max_depth=1, show_files=True)

        assert "script.py (2 lines)" in result
        assert "styles.css (3 lines)" in result
        assert "config.json (1 line)" in result
        assert "readme.md (4 lines)" in result
        assert "Total source code lines: 10" in result

    def test_no_line_count_for_non_source_files(self, tmp_path):
        """Test that non-source files don't show line counts"""
        # Create non-source files
        (tmp_path / "image.png").write_bytes(b'\x89PNG\r\n\x1a\n')
        (tmp_path / "data.bin").write_bytes(b'\x00\x01\x02\x03')

        result = list_directory_tree(str(tmp_path), max_depth=1, show_files=True)

        assert "image.png" in result
        assert " line)" not in result
        assert " lines)" not in result
        assert "Total source code lines:" not in result

    def test_line_count_in_nested_directories(self, tmp_path):
        """Test that line counts work in nested directories"""
        # Create nested structure with source files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("line1\nline2\nline3\n")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test.py").write_text("line1\nline2\n")

        result = list_directory_tree(str(tmp_path), max_depth=2, show_files=True)

        assert "main.py (3 lines)" in result
        assert "test.py (2 lines)" in result
        assert "Total source code lines: 5" in result

    def test_no_total_when_no_source_files(self, tmp_path):
        """Test that total is not shown when there are no source files"""
        # Create only non-source files
        (tmp_path / "image.png").write_bytes(b'\x89PNG\r\n\x1a\n')
        (tmp_path / "dir1").mkdir()

        result = list_directory_tree(str(tmp_path), max_depth=1, show_files=True)

        assert "Total source code lines:" not in result

    def test_singular_line_count(self, tmp_path):
        """Test that single line uses 'line' not 'lines'"""
        # Create a file with exactly 1 line
        (tmp_path / "single.py").write_text("single line\n")

        result = list_directory_tree(str(tmp_path), max_depth=1, show_files=True)

        assert "single.py (1 line)" in result
        assert "single.py (1 lines)" not in result

    def test_function_signatures_disabled_by_default(self, tmp_path):
        """Test that function signatures are not shown by default"""
        # Create a Python file with functions
        (tmp_path / "test.py").write_text(
            "def function1():\n"
            "    pass\n"
            "\n"
            "def function2(arg):\n"
            "    return arg\n"
        )

        result = list_directory_tree(str(tmp_path), max_depth=1, show_files=True)

        assert "test.py" in result
        # Should not contain function signatures when add_function_signatures is False (default)
        assert "def function1" not in result
        assert "def function2" not in result

    def test_function_signatures_python(self, tmp_path):
        """Test extracting Python function signatures in directory tree"""
        # Create Python files with functions
        (tmp_path / "module.py").write_text(
            "def hello_world():\n"
            '    print("Hello")\n'
            "\n"
            "class MyClass:\n"
            "    def method1(self, arg):\n"
            "        pass\n"
            "\n"
            "    def method2(self):\n"
            "        return True\n"
        )

        result = list_directory_tree(str(tmp_path), max_depth=1, show_files=True, add_function_signatures=True)

        assert "module.py" in result
        assert "def hello_world():" in result
        assert "class MyClass" in result
        assert "def method1(self, arg):" in result
        assert "def method2(self):" in result

    def test_function_signatures_multiple_languages(self, tmp_path):
        """Test extracting function signatures from multiple languages"""
        # Create Python file
        (tmp_path / "script.py").write_text(
            "def calculate(a, b):\n" "    return a + b\n"
        )

        # Create JavaScript file
        (tmp_path / "script.js").write_text(
            "function myFunction(param1, param2) {\n"
            "    return param1 + param2;\n"
            "}\n"
        )

        # Create PHP file
        (tmp_path / "script.php").write_text(
            "<?php\n"
            "class MyClass {\n"
            "    public function publicMethod($param) {\n"
            "        return $param;\n"
            "    }\n"
            "}\n"
            "?>"
        )

        result = list_directory_tree(str(tmp_path), max_depth=1, show_files=True, add_function_signatures=True)

        # Check Python signatures
        assert "script.py" in result
        assert "def calculate(a, b):" in result
        
        # Check JavaScript signatures
        assert "script.js" in result
        assert "function myFunction(param1, param2)" in result
        
        # Check PHP signatures
        assert "script.php" in result
        assert "class MyClass" in result
        assert "public function publicMethod($param)" in result

    def test_function_signatures_nested_directories(self, tmp_path):
        """Test extracting function signatures in nested directories"""
        # Create nested structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text(
            "def main():\n" '    print("Main function")\n'
        )

        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text(
            "def test_something():\n" "    assert True\n"
        )

        result = list_directory_tree(str(tmp_path), max_depth=2, show_files=True, add_function_signatures=True)

        assert "main.py" in result
        assert "def main():" in result
        assert "test_main.py" in result
        assert "def test_something():" in result

    def test_function_signatures_no_signatures_found(self, tmp_path):
        """Test handling files with no function signatures"""
        # Create a text file with no code
        (tmp_path / "readme.md").write_text(
            "# README\n" "This is a readme file with no code.\n"
        )

        result = list_directory_tree(str(tmp_path), max_depth=1, show_files=True, add_function_signatures=True)

        assert "readme.md" in result
        # Should not add error messages or "No function" messages to the tree
        assert "No function" not in result
        assert "Error:" not in result

    def test_function_signatures_with_non_source_files(self, tmp_path):
        """Test that non-source files don't attempt signature extraction"""
        # Create a binary file
        (tmp_path / "data.bin").write_bytes(b'\x00\x01\x02\x03')
        
        # Create a Python file
        (tmp_path / "script.py").write_text("def my_function():\n" "    pass\n")

        result = list_directory_tree(str(tmp_path), max_depth=1, show_files=True, add_function_signatures=True)

        # Binary file should be shown but no signatures attempted
        assert "data.bin" in result
        
        # Python file should show signatures
        assert "script.py" in result
        assert "def my_function():" in result

    def test_markdown_sections_skip_code_blocks(self, tmp_path):
        """Test that markdown section extraction skips content inside code blocks"""
        # Create a markdown file with headers inside code blocks
        (tmp_path / "doc.md").write_text(
            "# Title\n"
            "Some intro text\n"
            "```python\n"
            "# This is a Python comment, not a header\n"
            "def hello():\n"
            "    pass\n"
            "```\n"
            "## Section 2\n"
            "More text\n"
            "~~~bash\n"
            "# Bash comment\n"
            "echo hello\n"
            "~~~\n"
            "### Section 3\n"
        )

        result = list_directory_tree(str(tmp_path), max_depth=1, show_files=True, add_function_signatures=True)

        # Should show the file
        assert "doc.md" in result
        # Should show real markdown headers
        assert "# Title" in result
        assert "## Section 2" in result
        assert "### Section 3" in result
        # Should NOT show comments inside code blocks
        assert "# This is a Python comment" not in result
        assert "# Bash comment" not in result


class TestSearchInFiles:
    def test_basic_search(self, tmp_path):
        """Test basic text search"""
        file1 = tmp_path / "test1.py"
        file1.write_text("def my_function():\n    pass\n")

        result = search_in_files(str(tmp_path), "def my_function", file_extensions=('.py',))

        assert "test1.py" in result
        assert "def my_function" in result

    def test_case_sensitivity(self, tmp_path):
        """Test case-sensitive vs case-insensitive search"""
        file1 = tmp_path / "test.txt"
        file1.write_text("Hello World\n")

        # Case-insensitive (default)
        result = search_in_files(str(tmp_path), "hello")
        assert "Hello World" in result

        # Case-sensitive
        result = search_in_files(str(tmp_path), "hello", case_sensitive=True)
        assert "No matches found" in result

    def test_file_extension_filter(self, tmp_path):
        """Test file extension filtering"""
        (tmp_path / "test.py").write_text("python code")
        (tmp_path / "test.js").write_text("javascript code")

        result = search_in_files(str(tmp_path), "code", file_extensions=('.py',))
        assert "test.py" in result
        assert "test.js" not in result

    def test_max_results(self, tmp_path):
        """Test max results limiting"""
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text(f"match line {i}\n")

        result = search_in_files(str(tmp_path), "match", max_results=3)
        assert result.count("match line") == 3
        assert "stopped at 3 results" in result

    def test_no_matches(self, tmp_path):
        """Test when no matches are found"""
        (tmp_path / "test.txt").write_text("some content")

        result = search_in_files(str(tmp_path), "nonexistent")
        assert "No matches found" in result


class TestReadFileRange:
    def test_basic_read_range(self, tmp_path):
        """Test reading a byte range from a file"""
        file_path = tmp_path / "test.txt"
        content = "0123456789" * 10  # 100 bytes
        file_path.write_text(content)

        result = read_file_range(str(file_path), 10, 20)
        assert result == "0123456789"

    def test_invalid_range(self, tmp_path):
        """Test invalid byte ranges"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        with pytest.raises(ValueError):
            read_file_range(str(file_path), 20, 10)  # start > end

        with pytest.raises(ValueError):
            read_file_range(str(file_path), -1, 10)  # negative start

    def test_nonexistent_file(self):
        """Test reading from nonexistent file"""
        with pytest.raises(FileNotFoundError):
            read_file_range("/nonexistent/file.txt", 0, 10)


class TestGetFileInfo:
    def test_existing_file(self, tmp_path):
        """Test getting info for existing file"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test content")

        info = get_file_info(str(file_path))

        assert info['exists'] == True
        assert info['is_file'] == True
        assert info['is_dir'] == False
        assert info['size_bytes'] == 12  # "test content" is 12 bytes
        assert info['readable'] == True

    def test_directory(self, tmp_path):
        """Test getting info for directory"""
        dir_path = tmp_path / "testdir"
        dir_path.mkdir()

        info = get_file_info(str(dir_path))

        assert info['exists'] == True
        assert info['is_file'] == False
        assert info['is_dir'] == True

    def test_nonexistent_path(self):
        """Test getting info for nonexistent path"""
        info = get_file_info("/nonexistent/path")

        assert info['exists'] == False
        assert info['is_file'] == False
        assert info['is_dir'] == False


class TestListDirectory:
    def test_basic_listing(self, tmp_path):
        """Test basic directory listing"""
        (tmp_path / "file1.txt").write_text("content")
        (tmp_path / "file2.txt").write_text("content")
        (tmp_path / "dir1").mkdir()

        result = list_directory(str(tmp_path))

        assert len(result) == 3
        assert any("file1.txt" in r for r in result)
        assert any("file2.txt" in r for r in result)
        assert any("dir1" in r for r in result)

    def test_pattern_matching(self, tmp_path):
        """Test glob pattern matching"""
        (tmp_path / "test.py").write_text("python")
        (tmp_path / "test.js").write_text("javascript")

        result = list_directory(str(tmp_path), pattern="*.py")

        assert len(result) == 1
        assert "test.py" in result[0]

    def test_files_only(self, tmp_path):
        """Test files_only filter"""
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "dir").mkdir()

        result = list_directory(str(tmp_path), files_only=True)

        assert len(result) == 1
        assert "file.txt" in result[0]

    def test_dirs_only(self, tmp_path):
        """Test dirs_only filter"""
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "dir").mkdir()

        result = list_directory(str(tmp_path), dirs_only=True)

        assert len(result) == 1
        assert "dir" in result[0]

    def test_recursive(self, tmp_path):
        """Test recursive listing"""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("content")

        result = list_directory(str(tmp_path), pattern="*.txt", recursive=True)

        assert len(result) == 1
        assert "nested.txt" in result[0]


class TestMkdir:
    def test_create_directory(self, tmp_path):
        """Test directory creation"""
        new_dir = tmp_path / "newdir"

        result = mkdir(str(new_dir))

        assert result == True
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_create_nested_directories(self, tmp_path):
        """Test creating nested directories"""
        nested = tmp_path / "a" / "b" / "c"

        result = mkdir(str(nested), parents=True)

        assert result == True
        assert nested.exists()

    def test_existing_directory(self, tmp_path):
        """Test creating existing directory with parents=True"""
        new_dir = tmp_path / "existing"
        new_dir.mkdir()

        # Should not raise error with parents=True
        result = mkdir(str(new_dir), parents=True)
        assert result == True


class TestExtractFunctionSignatures:
    def test_python_functions(self, tmp_path):
        """Test extracting Python function signatures"""
        file_path = tmp_path / "test.py"
        file_path.write_text("""
def function1(arg1, arg2):
    pass

class MyClass:
    def method1(self):
        pass
""")

        result = extract_function_signatures(str(file_path), "python")

        assert "def function1(arg1, arg2):" in result
        assert "class MyClass" in result
        assert "def method1(self):" in result

    def test_php_object_oriented(self, tmp_path):
        """Test extracting PHP object-oriented signatures"""
        file_path = tmp_path / "test.php"
        file_path.write_text("""<?php
class MyClass {
    public function publicMethod($param1) {
        return $param1;
    }
    
    private function privateMethod() {
        return true;
    }
    
    protected static function staticMethod($a, $b) {
        return $a + $b;
    }
}

function standalone_function($param) {
    return $param;
}
?>""")

        result = extract_function_signatures(str(file_path), "php")

        assert "class MyClass" in result
        assert "public function publicMethod($param1)" in result
        assert "private function privateMethod()" in result
        assert "protected static function staticMethod($a, $b)" in result
        assert "function standalone_function($param)" in result

    def test_pascal_functions(self, tmp_path):
        """Test extracting Pascal/Object Pascal function signatures"""
        file_path = tmp_path / "test.pas"
        file_path.write_text("""program TestProgram;

type
  TMyClass = class
  private
    FValue: Integer;
  public
    constructor Create;
    destructor Destroy;
    function GetValue: Integer;
    procedure SetValue(AValue: Integer);
  end;

function CalculateSum(A, B: Integer): Integer;
begin
  Result := A + B;
end;

procedure PrintMessage(Msg: string);
begin
  WriteLn(Msg);
end;

constructor TMyClass.Create;
begin
  inherited;
  FValue := 0;
end;

destructor TMyClass.Destroy;
begin
  inherited;
end;

function TMyClass.GetValue: Integer;
begin
  Result := FValue;
end;

procedure TMyClass.SetValue(AValue: Integer);
begin
  FValue := AValue;
end;

begin
  PrintMessage('Hello, World!');
end.
""")

        result = extract_function_signatures(str(file_path), "pascal")

        assert "TMyClass = class" in result.lower() or "tmyclass = class" in result.lower()
        assert "function CalculateSum" in result or "function calculatesum" in result
        assert "procedure PrintMessage" in result or "procedure printmessage" in result
        assert "constructor Create" in result or "constructor create" in result
        assert "destructor Destroy" in result or "destructor destroy" in result

    def test_generic_fallback_with_function_keyword(self, tmp_path):
        """Test generic fallback for unsupported languages using 'function' keyword"""
        file_path = tmp_path / "test.lua"
        file_path.write_text("""
function myFunction(param1, param2)
    return param1 + param2
end

function anotherFunction()
    print("Hello")
end
""")

        result = extract_function_signatures(str(file_path), "lua")

        assert "function myFunction(param1, param2)" in result
        assert "function anotherFunction()" in result

    def test_generic_fallback_with_procedure_keyword(self, tmp_path):
        """Test generic fallback for unsupported languages using 'procedure' keyword"""
        file_path = tmp_path / "test.ada"
        file_path.write_text("""
procedure MyProcedure (Param1 : Integer) is
begin
    null;
end MyProcedure;

function MyFunction (A, B : Integer) return Integer is
begin
    return A + B;
end MyFunction;
""")

        result = extract_function_signatures(str(file_path), "ada")

        assert "procedure MyProcedure" in result
        assert "function MyFunction" in result

    def test_no_signatures_found(self, tmp_path):
        """Test when no signatures are found"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Just some plain text with no code")

        result = extract_function_signatures(str(file_path), "python")

        assert "No function/class signatures found" in result

    def test_markdown_sections_basic(self, tmp_path):
        """Test extracting markdown section headers"""
        file_path = tmp_path / "test.md"
        file_path.write_text("# Header 1\nSome text\n## Header 2\nMore text\n### Header 3\n")

        result = extract_function_signatures(str(file_path), "markdown")

        assert "# Header 1" in result
        assert "## Header 2" in result
        assert "### Header 3" in result
        assert "Some text" not in result

    def test_markdown_sections_skip_code_blocks(self, tmp_path):
        """Test that markdown section extraction skips content inside code blocks"""
        file_path = tmp_path / "test.md"
        file_path.write_text(
            "# Real Header\n"
            "```python\n"
            "# This is a comment in code block\n"
            "def hello():\n"
            "    pass\n"
            "```\n"
            "## Another Real Header\n"
            "~~~bash\n"
            "# Bash comment\n"
            "~~~\n"
            "### Third Header\n"
        )

        result = extract_function_signatures(str(file_path), "markdown")

        assert "# Real Header" in result
        assert "## Another Real Header" in result
        assert "### Third Header" in result
        # Comments inside code blocks should NOT be extracted
        assert "# This is a comment in code block" not in result
        assert "# Bash comment" not in result

    def test_markdown_no_sections(self, tmp_path):
        """Test markdown file with no section headers"""
        file_path = tmp_path / "test.md"
        file_path.write_text("Just plain text\nNo headers here\n")

        result = extract_function_signatures(str(file_path), "markdown")

        assert "No sections found" in result

    def test_markdown_mixed_delimiters(self, tmp_path):
        """Test that mixed code block delimiters are handled correctly"""
        file_path = tmp_path / "test.md"
        # In proper markdown, ``` must be closed with ``` and ~~~ with ~~~
        # So a ~~~ should NOT close a block opened with ```
        file_path.write_text(
            "# Header 1\n"
            "```python\n"
            "# comment in code\n"
            "~~~\n"  # This should NOT close the block
            "## Should be skipped (still in code block)\n"
        )

        result = extract_function_signatures(str(file_path), "markdown")

        # Only the first header should be extracted since the code block is never properly closed
        assert "# Header 1" in result
        assert "## Should be skipped" not in result

    def test_nonexistent_file(self):
        """Test nonexistent file"""
        result = extract_function_signatures("/nonexistent/file.py", "python")
        assert "not found" in result


class TestCompareFiles:
    def test_identical_files(self, tmp_path):
        """Test comparing identical files"""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("same content\n")
        file2.write_text("same content\n")

        result = compare_files(str(file1), str(file2))

        assert "identical" in result.lower()

    def test_different_files(self, tmp_path):
        """Test comparing different files"""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("line 1\nline 2\n")
        file2.write_text("line 1\nline 2 modified\n")

        result = compare_files(str(file1), str(file2))

        assert "---" in result  # Unified diff format
        assert "+++" in result
        assert "line 2" in result or "modified" in result

    def test_nonexistent_file(self, tmp_path):
        """Test comparing with nonexistent file"""
        file1 = tmp_path / "exists.txt"
        file1.write_text("content")

        result = compare_files(str(file1), "/nonexistent/file.txt")
        assert "not found" in result.lower()


class TestCompareFolders:
    def test_identical_folders(self, tmp_path):
        """Test comparing identical folders"""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        # Create identical source files
        (folder1 / "test.py").write_text("def hello():\n    pass\n")
        (folder2 / "test.py").write_text("def hello():\n    pass\n")
        
        result = compare_folders(str(folder1), str(folder2))
        
        assert "identical" in result.lower()

    def test_folders_with_different_files(self, tmp_path):
        """Test comparing folders with different content"""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        # Create different content
        (folder1 / "test.py").write_text("def hello():\n    pass\n")
        (folder2 / "test.py").write_text("def hello():\n    print('hi')\n")
        
        result = compare_folders(str(folder1), str(folder2))
        
        assert "DIFFERENCES" in result
        assert "test.py" in result

    def test_folders_with_unique_files(self, tmp_path):
        """Test comparing folders where each has unique files"""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        # Files only in folder1
        (folder1 / "only_in_1.py").write_text("code here")
        
        # Files only in folder2
        (folder2 / "only_in_2.js").write_text("code here")
        
        result = compare_folders(str(folder1), str(folder2))
        
        assert "ONLY IN FOLDER 1" in result
        assert "only_in_1.py" in result
        assert "ONLY IN FOLDER 2" in result
        assert "only_in_2.js" in result

    def test_folders_with_nested_structure(self, tmp_path):
        """Test comparing folders with nested directory structure"""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        # Create nested structure
        (folder1 / "subdir").mkdir()
        (folder2 / "subdir").mkdir()
        
        (folder1 / "subdir" / "nested.py").write_text("def func1():\n    pass\n")
        (folder2 / "subdir" / "nested.py").write_text("def func2():\n    pass\n")
        
        result = compare_folders(str(folder1), str(folder2))
        
        assert "subdir" in result or "nested.py" in result
        assert "DIFFERENCES" in result

    def test_only_source_code_files_compared(self, tmp_path):
        """Test that only source code files are compared"""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        # Create source code files (should be compared)
        (folder1 / "test.py").write_text("python code")
        (folder2 / "test.py").write_text("different python")
        
        # Create binary files (should be ignored)
        (folder1 / "data.bin").write_bytes(b'\x00\x01\x02')
        (folder2 / "data.bin").write_bytes(b'\xff\xfe\xfd')
        
        result = compare_folders(str(folder1), str(folder2))
        
        # Should report difference in .py file
        assert "test.py" in result
        # Should NOT mention .bin file
        assert "data.bin" not in result

    def test_nonexistent_folder(self, tmp_path):
        """Test comparing with nonexistent folder"""
        folder1 = tmp_path / "exists"
        folder1.mkdir()
        
        result = compare_folders(str(folder1), "/nonexistent/folder")
        assert "not found" in result.lower()

    def test_empty_folders(self, tmp_path):
        """Test comparing empty folders"""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        result = compare_folders(str(folder1), str(folder2))
        
        assert "identical" in result.lower()

    def test_mixed_scenario(self, tmp_path):
        """Test a complex scenario with identical, different, and unique files"""
        folder1 = tmp_path / "folder1"
        folder2 = tmp_path / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        # Identical file
        (folder1 / "same.py").write_text("same content")
        (folder2 / "same.py").write_text("same content")
        
        # Different file
        (folder1 / "diff.js").write_text("version 1")
        (folder2 / "diff.js").write_text("version 2")
        
        # Unique files
        (folder1 / "unique1.html").write_text("only in 1")
        (folder2 / "unique2.css").write_text("only in 2")
        
        result = compare_folders(str(folder1), str(folder2))
        
        assert "SUMMARY" in result
        assert "same.py" not in result or "Identical files: 1" in result
        assert "diff.js" in result
        assert "unique1.html" in result
        assert "unique2.css" in result


class TestDeleteOperations:
    def test_delete_file(self, tmp_path):
        """Test file deletion"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        result = delete_file(str(file_path))

        assert result == True
        assert not file_path.exists()

    def test_delete_nonexistent_file(self):
        """Test deleting nonexistent file"""
        with pytest.raises(FileNotFoundError):
            delete_file("/nonexistent/file.txt")

    def test_delete_directory_as_file(self, tmp_path):
        """Test trying to delete directory with delete_file"""
        dir_path = tmp_path / "dir"
        dir_path.mkdir()

        with pytest.raises(IsADirectoryError):
            delete_file(str(dir_path))

    def test_delete_empty_directory(self, tmp_path):
        """Test deleting empty directory"""
        dir_path = tmp_path / "emptydir"
        dir_path.mkdir()

        result = delete_directory(str(dir_path), recursive=False)

        assert result == True
        assert not dir_path.exists()

    def test_delete_directory_recursive(self, tmp_path):
        """Test deleting directory with contents"""
        dir_path = tmp_path / "fulldir"
        dir_path.mkdir()
        (dir_path / "file.txt").write_text("content")
        (dir_path / "subdir").mkdir()

        result = delete_directory(str(dir_path), recursive=True)

        assert result == True
        assert not dir_path.exists()


class TestCountLinesOfCode:
    def test_basic_count(self, tmp_path):
        """Test basic line counting"""
        (tmp_path / "test.py").write_text("line1\nline2\nline3\n")
        (tmp_path / "test.js").write_text("line1\nline2\n")

        result = count_lines_of_code(str(tmp_path), file_extensions=('.py', '.js'))

        assert result['.py'] == 3
        assert result['.js'] == 2
        assert result['_total'] == 5

    def test_empty_directory(self, tmp_path):
        """Test counting in empty directory"""
        result = count_lines_of_code(str(tmp_path))

        assert result['_total'] == 0

    def test_invalid_directory(self):
        """Test counting in nonexistent directory"""
        result = count_lines_of_code("/nonexistent/path")

        assert 'error' in result


class TestReadFirstNLines:
    def test_basic_read_first_lines(self, tmp_path):
        """Test reading the first n lines of a file"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")

        result = read_first_n_lines(str(file_path), 3)

        assert result == "line 1\nline 2\nline 3\n"

    def test_n_exceeds_file_length(self, tmp_path):
        """Test reading when n exceeds the number of lines in file"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\n")

        result = read_first_n_lines(str(file_path), 10)

        assert result == "line 1\nline 2\n"

    def test_read_single_line(self, tmp_path):
        """Test reading exactly one line"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\nline 3\n")

        result = read_first_n_lines(str(file_path), 1)

        assert result == "line 1\n"

    def test_nonexistent_file(self):
        """Test reading from nonexistent file"""
        with pytest.raises(FileNotFoundError):
            read_first_n_lines("/nonexistent/file.txt", 5)

    def test_n_less_than_one(self, tmp_path):
        """Test that n < 1 raises ValueError"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\n")

        with pytest.raises(ValueError):
            read_first_n_lines(str(file_path), 0)

        with pytest.raises(ValueError):
            read_first_n_lines(str(file_path), -1)


class TestReadLastNLines:
    def test_basic_read_last_lines(self, tmp_path):
        """Test reading the last n lines of a file"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")

        result = read_last_n_lines(str(file_path), 3)

        assert result == "line 3\nline 4\nline 5\n"

    def test_n_exceeds_file_length(self, tmp_path):
        """Test reading when n exceeds the number of lines in file"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\n")

        result = read_last_n_lines(str(file_path), 10)

        assert result == "line 1\nline 2\n"

    def test_read_single_line(self, tmp_path):
        """Test reading exactly one line from the end"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\nline 3\n")

        result = read_last_n_lines(str(file_path), 1)

        assert result == "line 3\n"

    def test_nonexistent_file(self):
        """Test reading from nonexistent file"""
        with pytest.raises(FileNotFoundError):
            read_last_n_lines("/nonexistent/file.txt", 5)

    def test_n_less_than_one(self, tmp_path):
        """Test that n < 1 raises ValueError"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\n")

        with pytest.raises(ValueError):
            read_last_n_lines(str(file_path), 0)

        with pytest.raises(ValueError):
            read_last_n_lines(str(file_path), -1)


class TestDeleteLinesFromFile:
    def test_delete_single_line(self, tmp_path):
        """Test deleting a single line from a file"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\nline 3\nline 4\n")

        result = delete_lines_from_file(str(file_path), 2)

        assert result == "line 1\nline 3\nline 4\n"
        assert file_path.read_text() == "line 1\nline 3\nline 4\n"

    def test_delete_line_range(self, tmp_path):
        """Test deleting a range of lines from a file"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")

        result = delete_lines_from_file(str(file_path), 2, 4)

        assert result == "line 1\nline 5\n"
        assert file_path.read_text() == "line 1\nline 5\n"

    def test_delete_first_line(self, tmp_path):
        """Test deleting the first line"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\nline 3\n")

        result = delete_lines_from_file(str(file_path), 1)

        assert result == "line 2\nline 3\n"

    def test_delete_last_line(self, tmp_path):
        """Test deleting the last line"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\nline 3\n")

        result = delete_lines_from_file(str(file_path), 3)

        assert result == "line 1\nline 2\n"

    def test_delete_all_lines(self, tmp_path):
        """Test deleting all lines"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\nline 3\n")

        result = delete_lines_from_file(str(file_path), 1, 3)

        assert result == ""
        assert file_path.read_text() == ""

    def test_nonexistent_file(self):
        """Test deleting from nonexistent file"""
        with pytest.raises(FileNotFoundError):
            delete_lines_from_file("/nonexistent/file.txt", 1)

    def test_start_line_less_than_one(self, tmp_path):
        """Test that start_line < 1 raises ValueError"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\n")

        with pytest.raises(ValueError):
            delete_lines_from_file(str(file_path), 0)

        with pytest.raises(ValueError):
            delete_lines_from_file(str(file_path), -1)

    def test_end_line_less_than_start_line(self, tmp_path):
        """Test that end_line < start_line raises ValueError"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\nline 3\n")

        with pytest.raises(ValueError):
            delete_lines_from_file(str(file_path), 3, 1)

    def test_start_line_beyond_file_length(self, tmp_path):
        """Test that start_line beyond file length raises ValueError"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\n")

        with pytest.raises(ValueError):
            delete_lines_from_file(str(file_path), 10)

    def test_end_line_exceeds_file_length(self, tmp_path):
        """Test that end_line exceeding file length is capped"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("line 1\nline 2\nline 3\n")

        result = delete_lines_from_file(str(file_path), 2, 10)

        assert result == "line 1\n"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
