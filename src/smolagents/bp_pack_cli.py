# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

"""CLI wrapper for source_code_to_string."""

import argparse

from smolagents.bp_tools import source_code_to_string, DEFAULT_SOURCE_CODE_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(
        prog="bppack",
        description="Pack a folder's source code into a single string with XML-like file tags.",
    )
    parser.add_argument(
        "folder",
        help="root folder to scan",
    )
    parser.add_argument(
        "-e", "--extensions",
        type=lambda s: tuple(ext if ext.startswith(".") else f".{ext}" for ext in s.split(",")),
        default=DEFAULT_SOURCE_CODE_EXTENSIONS,
        help="comma-separated file extensions to include (default: common source extensions)",
    )
    parser.add_argument(
        "--strip-pascal-comments",
        action="store_true",
        help="remove Pascal comments from .pas/.inc files",
    )
    parser.add_argument(
        "--exclude",
        type=lambda s: tuple(s.split(",")),
        default=("excluded_folder", "excluded_file.pas"),
        help="comma-separated list of files/folders to exclude",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="write output to a file instead of stdout",
    )

    args = parser.parse_args()

    result = source_code_to_string(
        folder_name=args.folder,
        allowed_extensions=args.extensions,
        remove_pascal_comments=args.strip_pascal_comments,
        exclude_list=args.exclude,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Packed output written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
