"""CLI wrapper for extract_function_signatures."""

import argparse

from smolagents.bp_tools import extract_function_signatures, detect_language


def main():
    parser = argparse.ArgumentParser(
        prog="bpsig",
        description="Extract function and class signatures from a source file without the full implementation.",
    )
    parser.add_argument("file", help="source code file to extract signatures from")
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="programming language (auto-detected from extension if omitted)",
    )

    args = parser.parse_args()

    language = args.language or detect_language(args.file)
    result = extract_function_signatures(args.file, language=language)
    print(result)


if __name__ == "__main__":
    main()
