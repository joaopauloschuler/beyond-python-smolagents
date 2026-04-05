"""CLI wrapper for string_to_source_code."""

import argparse
import sys

from smolagents.bp_tools import string_to_source_code


def main():
    parser = argparse.ArgumentParser(
        prog="bpunpack",
        description="Reconstruct source files from a packed string (produced by bppack).",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="packed file to read (default: stdin)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="base directory to write files into (default: current directory)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="skip files that already exist instead of overwriting",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="suppress status messages",
    )

    args = parser.parse_args()

    if args.input == "-":
        content = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            content = f.read()

    string_to_source_code(
        string_with_files=content,
        output_base_dir=args.output_dir,
        overwrite=not args.no_overwrite,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
