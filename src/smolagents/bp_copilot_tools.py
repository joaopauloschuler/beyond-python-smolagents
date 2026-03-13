"""GitHub Copilot SDK tool wrappers for bpsa helpers.

Wraps bp_tools functions using the Copilot SDK's ``define_tool`` decorator
so they can be passed directly to ``SessionConfig["tools"]``.

Requires the ``github-copilot-sdk`` package (``pip install github-copilot-sdk``).

Usage::

    from smolagents.bp_copilot_tools import ALL_COPILOT_TOOLS

    session = await client.create_session({
        "tools": ALL_COPILOT_TOOLS,
        ...
    })
"""

from pydantic import BaseModel, Field

from copilot.tools import define_tool
from smolagents.bp_tools import (
    compare_files as _compare_files,
    compare_folders as _compare_folders,
    count_lines_of_code as _count_lines_of_code,
    detect_language,
    extract_function_signatures,
    inject_tree as _inject_tree,
    list_directory_tree as _list_directory_tree,
    search_in_files as _search_in_files,
)


# ---------------------------------------------------------------------------
# Param models
# ---------------------------------------------------------------------------


class DirectoryTreeParams(BaseModel):
    folder_path: str = Field(description="Root folder path to visualize")
    max_depth: int = Field(default=6, description="Maximum depth to traverse")
    show_signatures: bool = Field(
        default=False,
        description="Whether to extract and display function/class signatures",
    )


class ShowSignaturesParams(BaseModel):
    file_path: str = Field(description="Path to the source code file")


class SearchInFilesParams(BaseModel):
    folder_path: str = Field(description="Root folder to search in, or path to a single file")
    search_pattern: str = Field(description="Text pattern to search for")
    file_extensions: list[str] | None = Field(
        default=None,
        description="Optional list of file extensions to search (e.g. ['.py', '.js']). If None, searches all text files",
    )
    case_sensitive: bool = Field(default=False, description="Whether the search is case-sensitive")
    max_results: int = Field(default=50, description="Maximum number of results to return")


class InjectTreeParams(BaseModel):
    folder: str = Field(description="Path to the folder to generate the tree from")


class CompareFilesParams(BaseModel):
    file1: str = Field(description="Path to the first file")
    file2: str = Field(description="Path to the second file")
    context_lines: int = Field(default=3, description="Number of context lines around differences")


class CompareFoldersParams(BaseModel):
    folder1: str = Field(description="Path to the first folder")
    folder2: str = Field(description="Path to the second folder")
    context_lines: int = Field(default=3, description="Number of context lines around differences")


class CountLinesOfCodeParams(BaseModel):
    folder_path: str = Field(description="Root folder to analyze")
    file_extensions: list[str] = Field(
        default=['.py', '.js', '.java', '.cpp', '.c', '.php', '.rb'],
        description="File extensions to count",
    )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@define_tool(description=(
    "Show a tree view of a directory structure with file line counts. "
    "Optionally extracts function and class signatures from source files. "
    "Useful for understanding project layout without reading full files."
))
def list_directory_tree(params: DirectoryTreeParams) -> str:
    return _list_directory_tree(
        folder_path=params.folder_path,
        max_depth=params.max_depth,
        add_function_signatures=params.show_signatures,
    )


@define_tool(description=(
    "Extract function and class signatures from a source file without loading "
    "the full implementation. Supports Python, JavaScript/TypeScript, Java, "
    "PHP, C/C++, Markdown (section headers), and a generic fallback."
))
def show_signatures(params: ShowSignaturesParams) -> str:
    language = detect_language(params.file_path)
    result = extract_function_signatures(params.file_path, language)
    if not result:
        return f"No signatures found in '{params.file_path}'"
    return result


@define_tool(description=(
    "Search for a text pattern in files within a folder (and subfolders) or a single file. "
    "Returns matching lines with file paths and line numbers. More ergonomic than grep."
))
def search_in_files(params: SearchInFilesParams) -> str:
    exts = tuple(params.file_extensions) if params.file_extensions else None
    return _search_in_files(
        folder_path=params.folder_path,
        search_pattern=params.search_pattern,
        file_extensions=exts,
        case_sensitive=params.case_sensitive,
        max_results=params.max_results,
    )


@define_tool(description=(
    "Generate a directory tree with function signatures and guidance text. "
    "Useful as context injection at the start of tasks to understand project structure."
))
def inject_tree(params: InjectTreeParams) -> str:
    return _inject_tree(folder=params.folder)


@define_tool(description=(
    "Compare two files and show differences in unified diff format."
))
def compare_files(params: CompareFilesParams) -> str:
    return _compare_files(
        file1=params.file1,
        file2=params.file2,
        context_lines=params.context_lines,
    )


@define_tool(description=(
    "Compare two folders and show differences for all source code files. "
    "Reports files unique to each folder and unified diffs for changed files."
))
def compare_folders(params: CompareFoldersParams) -> str:
    return _compare_folders(
        folder1=params.folder1,
        folder2=params.folder2,
        context_lines=params.context_lines,
    )


@define_tool(description=(
    "Count lines of code in a project, broken down by file type. "
    "Helps understand project size and composition."
))
def count_lines_of_code(params: CountLinesOfCodeParams) -> dict:
    return _count_lines_of_code(
        folder_path=params.folder_path,
        file_extensions=tuple(params.file_extensions),
    )


ALL_COPILOT_TOOLS = [
    list_directory_tree,
    show_signatures,
    search_in_files,
    inject_tree,
    compare_files,
    compare_folders,
    count_lines_of_code,
]
