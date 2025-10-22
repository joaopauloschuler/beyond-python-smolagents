#!/usr/bin/env python3
"""
Unit tests for context-efficient tools in bp_tools.py
"""

import os
import sys
import pytest
import tempfile
import shutil

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smolagents.bp_tools import (
    list_directory_tree,
    search_in_files,
    read_file_range,
    get_file_info,
    list_directory,
    mkdir,
    extract_function_signatures,
    compare_files,
    delete_file,
    delete_directory,
    count_lines_of_code
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
    
    def test_unsupported_language(self, tmp_path):
        """Test unsupported language"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")
        
        result = extract_function_signatures(str(file_path), "cobol")
        assert "not supported" in result
    
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
