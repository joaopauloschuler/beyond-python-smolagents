"""CLI wrapper for list_directory_tree."""

import argparse
import sys

from smolagents.bp_tools import list_directory_tree


def main():
    parser = argparse.ArgumentParser(
        prog="bptree",
        description="Display a tree view of a directory with line counts and optional function signatures.",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="root folder to visualize (default: current directory)",
    )
    parser.add_argument(
        "-d", "--depth",
        type=int,
        default=6,
        help="maximum depth to traverse (default: 6)",
    )
    parser.add_argument(
        "--no-files",
        action="store_true",
        help="show only directories, hide files",
    )
    parser.add_argument(
        "-s", "--signatures",
        action="store_true",
        help="extract and display function signatures for source files",
    )
    parser.add_argument(
        "--skip-dirs",
        type=lambda s: s.split(","),
        default=None,
        help="comma-separated list of directory names to skip (shown but not traversed)",
    )

    args = parser.parse_args()

    result = list_directory_tree(
        folder_path=args.folder,
        max_depth=args.depth,
        show_files=not args.no_files,
        add_function_signatures=args.signatures,
        skip_dirs=args.skip_dirs,
    )
    print(result)


if __name__ == "__main__":
    main()
