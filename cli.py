#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from utils.tools import LanguageTools


def read_code_from_file(file_path: str) -> str:
    """
    Read code from a file.

    Args:
        file_path: Path to the file

    Returns:
        The file contents as a string
    """
    with open(file_path, "r") as f:
        return f.read()


def read_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """
    Read test cases from a JSON file.

    Args:
        file_path: Path to the test cases file

    Returns:
        List of test cases
    """
    with open(file_path, "r") as f:
        return json.load(f)


def print_test_results(results: List[Dict[str, Any]]) -> None:
    """
    Print test results in a formatted way.

    Args:
        results: List of test results
    """
    total_tests = len(results)
    passed_tests = sum(1 for result in results if result["passed"])

    print(f"\nTEST RESULTS: {passed_tests}/{total_tests} passed\n")
    print("=" * 80)

    for i, result in enumerate(results):
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"Test Case {i + 1}: {status}")
        print(f"Input: {result['input']}")
        print(f"Expected: {result['expected']}")
        print(f"Actual: {result['output']}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run and test code in different programming languages."
    )
    parser.add_argument("source", help="Source code file path")
    parser.add_argument("--language", "-l", required=True, help="Programming language")
    parser.add_argument(
        "--config",
        "-c",
        default="language_tools.yaml",
        help="Language tools configuration file",
    )
    parser.add_argument("--run", "-r", action="store_true", help="Run the code")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the code only (for compiled languages)",
    )
    parser.add_argument("--test", "-t", help="Path to test cases JSON file")
    parser.add_argument(
        "--args", nargs="+", help="Command line arguments for the program"
    )

    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: Source file '{args.source}' not found.")
        sys.exit(1)

    # Create LanguageTools instance
    tools = LanguageTools(args.config)

    # Read the source code
    code = read_code_from_file(args.source)

    # Compile only
    if args.compile:
        success, message = tools.compile(
            code, args.language, filename=os.path.basename(args.source)
        )
        print(message)
        if not success:
            sys.exit(1)
        sys.exit(0)

    # Run with test cases
    if args.test:
        if not os.path.exists(args.test):
            print(f"Error: Test cases file '{args.test}' not found.")
            sys.exit(1)

        test_cases = read_test_cases(args.test)
        results = tools.test(code, args.language, test_cases)
        print_test_results(results)

        # Exit with non-zero status if any test failed
        if any(not result["passed"] for result in results):
            sys.exit(1)
        sys.exit(0)

    # Run the code
    if args.run:
        success, output = tools.run(
            code, args.language, args=args.args, filename=os.path.basename(args.source)
        )

        if not success:
            print(f"Error: {output}")
            sys.exit(1)

        print(output)
        sys.exit(0)

    # If no action specified, print usage
    parser.print_help()


if __name__ == "__main__":
    main()
