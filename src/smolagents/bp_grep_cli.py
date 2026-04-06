# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

"""CLI wrapper for search_in_files."""

import argparse

from smolagents.bp_tools import search_in_files


def main():
    parser = argparse.ArgumentParser(
        prog="bpgrep",
        description="Search for a text pattern in files within a folder or a single file.",
    )
    parser.add_argument("pattern", help="text pattern to search for")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="folder or file to search in (default: current directory)",
    )
    parser.add_argument(
        "-e", "--extensions",
        type=lambda s: tuple(ext if ext.startswith(".") else f".{ext}" for ext in s.split(",")),
        default=None,
        help="comma-separated file extensions to search (e.g. py,js,ts)",
    )
    parser.add_argument(
        "-c", "--case-sensitive",
        action="store_true",
        help="make the search case-sensitive (default: case-insensitive)",
    )
    parser.add_argument(
        "-m", "--max-results",
        type=int,
        default=50,
        help="maximum number of results (default: 50)",
    )

    args = parser.parse_args()

    result = search_in_files(
        folder_path=args.path,
        search_pattern=args.pattern,
        file_extensions=args.extensions,
        case_sensitive=args.case_sensitive,
        max_results=args.max_results,
    )
    print(result)


if __name__ == "__main__":
    main()
