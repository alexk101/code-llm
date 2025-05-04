#!/usr/bin/env python3
"""
Script to run linting and formatting tools on the codebase.
Run with `python lint.py` to format and lint all files.
Run with `python lint.py --check` to only check for issues without making changes.
"""

import argparse
import subprocess
import sys


def run_command(cmd):
    """Run a command and return success/failure."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run linting and formatting tools")
    parser.add_argument(
        "--check", action="store_true", help="Only check, don't modify files"
    )
    args = parser.parse_args()

    success = True

    # Run autoflake
    autoflake_cmd = ["autoflake"]
    if not args.check:
        autoflake_cmd.append("--in-place")
    else:
        autoflake_cmd.append("--check")
    autoflake_cmd.extend(["."])
    success = run_command(autoflake_cmd) and success

    # Run ruff format
    ruff_format_cmd = ["ruff", "format"]
    if args.check:
        ruff_format_cmd.append("--check")
    ruff_format_cmd.extend(["."])
    success = run_command(ruff_format_cmd) and success

    # Run ruff lint
    ruff_lint_cmd = ["ruff", "check"]
    if not args.check:
        ruff_lint_cmd.append("--fix")
    ruff_lint_cmd.extend(["."])
    success = run_command(ruff_lint_cmd) and success

    if not success:
        print("\nLinting found issues.")
        sys.exit(1)
    else:
        print("\nAll linting checks passed!")


if __name__ == "__main__":
    main()
