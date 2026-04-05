# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

"""CLI wrapper for count_lines_of_code."""

import argparse

from smolagents.bp_tools import count_lines_of_code


def main():
    parser = argparse.ArgumentParser(
        prog="bploc",
        description="Count lines of code in a project, broken down by file type.",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="root folder to analyze (default: current directory)",
    )
    parser.add_argument(
        "-e", "--extensions",
        type=lambda s: tuple(ext if ext.startswith(".") else f".{ext}" for ext in s.split(",")),
        default=(".py", ".js", ".java", ".cpp", ".c", ".php", ".rb"),
        help="comma-separated file extensions to count (default: py,js,java,cpp,c,php,rb)",
    )

    args = parser.parse_args()

    result = count_lines_of_code(
        folder_path=args.folder,
        file_extensions=args.extensions,
    )

    if "error" in result:
        print(result["error"])
        return

    total = result.pop("_total", 0)

    if not result:
        print("No source files found.")
        return

    max_ext_len = max(len(ext) for ext in result)
    for ext in sorted(result, key=result.get, reverse=True):
        print(f"  {ext:<{max_ext_len}}  {result[ext]:>8} lines")
    print(f"  {'total':<{max_ext_len}}  {total:>8} lines")


if __name__ == "__main__":
    main()
