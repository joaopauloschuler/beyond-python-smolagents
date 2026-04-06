# BPSA - Beyond Python SmolAgents
# https://github.com/joaopauloschuler/beyond-python-smolagents
#
# Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and others.
# Refer to the git commit history for individual authorship.
# Licensed under the Apache License, Version 2.0

"""CLI wrapper for pascal_interface_to_string."""

import argparse

from smolagents.bp_tools import pascal_interface_to_string


def main():
    parser = argparse.ArgumentParser(
        prog="bppas",
        description="Extract Pascal unit interface sections from a folder of source files.",
    )
    parser.add_argument(
        "folder",
        help="root folder to scan for Pascal files",
    )
    parser.add_argument(
        "--strip-comments",
        action="store_true",
        help="remove Pascal comments from the extracted interfaces",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="write output to a file instead of stdout",
    )

    args = parser.parse_args()

    result = pascal_interface_to_string(
        folder_name=args.folder,
        remove_pascal_comments=args.strip_comments,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Output written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
