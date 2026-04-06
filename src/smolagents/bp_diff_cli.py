# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

"""CLI wrapper for compare_files and compare_folders."""

import argparse
import os

from smolagents.bp_tools import compare_files, compare_folders


def main():
    parser = argparse.ArgumentParser(
        prog="bpdiff",
        description="Compare two files or two folders and show differences in unified diff format.",
    )
    parser.add_argument("path1", help="first file or folder")
    parser.add_argument("path2", help="second file or folder")
    parser.add_argument(
        "-c", "--context",
        type=int,
        default=3,
        help="number of context lines around differences (default: 3)",
    )

    args = parser.parse_args()

    if os.path.isfile(args.path1) and os.path.isfile(args.path2):
        result = compare_files(args.path1, args.path2, context_lines=args.context)
    elif os.path.isdir(args.path1) and os.path.isdir(args.path2):
        result = compare_folders(args.path1, args.path2, context_lines=args.context)
    else:
        result = "Error: both paths must be files or both must be directories"

    print(result)


if __name__ == "__main__":
    main()
