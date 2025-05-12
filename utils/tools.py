import logging
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import yaml

from utils.definitions import TIMEOUT

logger = logging.getLogger(__name__)


def convert_python2_print_to_python3(code: str) -> str:
    """
    Convert Python 2 style print statements to Python 3 format.

    Args:
        code: Python source code that may contain Python 2 style print statements

    Returns:
        Code with print statements converted to Python 3 format
    """
    # Regular expression to match Python 2 style print statements
    # This matches "print" followed by a space and content,
    # but not if it's already in parentheses
    # Also handles trailing commas for newline suppression
    pattern = r"(^|\n|\s)print\s+([^(].*?)($|\n)"

    def replace_print(match):
        prefix = match.group(1)
        content = match.group(2).rstrip()
        suffix = match.group(3)

        # Handle trailing comma for newline suppression
        if content.endswith(","):
            return f"{prefix}print({content[:-1]}, end=''){suffix}"
        else:
            return f"{prefix}print({content}){suffix}"

    # Replace print statements in the code
    return re.sub(pattern, replace_print, code)


def normalize_indentation(code: str) -> str:
    """
    Normalize indentation in Python code by converting tabs to spaces
    and ensuring consistent indentation.

    Args:
        code: Python source code that may have inconsistent indentation

    Returns:
        Code with normalized indentation
    """
    # Split the code into lines
    lines = code.split("\n")
    normalized_lines = []

    # Process each line
    for line in lines:
        # Convert tabs to 4 spaces (Python standard)
        normalized_line = line.replace("\t", "    ")
        normalized_lines.append(normalized_line)

    # Join the lines back together
    return "\n".join(normalized_lines)


def sanitize_code(code: str, language: str = None) -> str:
    """
    Sanitize code by removing non-printable characters and normalizing whitespace.

    Args:
        code: Source code to sanitize
        language: Programming language of the code (optional)

    Returns:
        Sanitized code
    """
    # Replace non-breaking spaces (U+00A0) with regular spaces
    code = code.replace("\u00a0", " ")

    # Replace other common problematic characters
    code = code.replace("\u2018", "'")  # Left single quotation mark
    code = code.replace("\u2019", "'")  # Right single quotation mark
    code = code.replace("\u201c", '"')  # Left double quotation mark
    code = code.replace("\u201d", '"')  # Right double quotation mark
    code = code.replace("\u2013", "-")  # En dash
    code = code.replace("\u2014", "--")  # Em dash

    # Remove any other non-ASCII characters
    code = "".join(c if ord(c) < 128 else " " for c in code)

    # Normalize line endings to use just \n
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    # Python-specific sanitization
    if language and language.lower() == "python":
        # Convert Python 2 print statements to Python 3
        code = convert_python2_print_to_python3(code)

        # Normalize indentation (convert tabs to spaces)
        code = normalize_indentation(code)

    return code


class LanguageTools:
    """
    Provides language-specific tools for compiling and running code.
    Configuration is loaded from a YAML file that specifies commands
    for different programming languages.
    """

    def __init__(self, config_path: str = "language_tools.yaml"):
        """
        Initialize the LanguageTools with the given configuration file.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the language configuration from the YAML file.

        Returns:
            Dictionary containing language configurations
        """
        if not os.path.exists(self.config_path):
            logger.warning(
                f"Config file {self.config_path} not found. Using empty configuration."
            )
            return {}

        with open(self.config_path, "r") as f:
            try:
                config = yaml.safe_load(f)

                # Post-process the config to ensure run commands are strings
                for lang, lang_config in config.items():
                    if "run" in lang_config and not isinstance(lang_config["run"], str):
                        # If run is a dictionary, convert it to a string
                        if isinstance(lang_config["run"], dict):
                            logger.warning(
                                f"Fixed run command for {lang} from dict to string"
                            )
                            # For compiled languages, the run command is
                            # typically just the output path
                            lang_config["run"] = "{output}"

                return config
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML configuration: {e}")
                return {}

    def get_language_config(self, language: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific language.

        Args:
            language: The programming language name

        Returns:
            Dictionary containing the language's configuration
        """
        language = language.lower()
        return self.config.get(language, {})

    def compile(
        self, code: str, language: str, filename: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Compile the given code using the language-specific compiler.

        Args:
            code: The source code to compile
            language: The programming language
            filename: Optional filename to use
                (default is temp file with appropriate extension)

        Returns:
            Tuple of (success, output/error message)
        """
        language = language.lower()
        config = self.get_language_config(language)

        if not config:
            return False, f"No configuration found for language: {language}"

        compile_cmd = config.get("compile")
        if not compile_cmd:
            # If no compile command is specified, assume the language is interpreted
            return True, "Language does not require compilation"

        # Sanitize code to remove non-printable characters
        code = sanitize_code(code, language)

        # Get file extension for this language
        extension = config.get("extension", language)

        # Create a temporary file with the code
        if filename:
            temp_dir = tempfile.mkdtemp()
            source_path = os.path.join(temp_dir, filename)
        else:
            fd, source_path = tempfile.mkstemp(suffix=f".{extension}")
            os.close(fd)

        with open(source_path, "w") as f:
            f.write(code)

        # Replace placeholders in the compile command
        compile_cmd = compile_cmd.replace("{source}", source_path)
        output_path = source_path.rsplit(".", 1)[0]
        compile_cmd = compile_cmd.replace("{output}", output_path)

        # Run the compile command
        logger.info(f"Running compile command: {compile_cmd}")
        result = subprocess.run(
            compile_cmd, shell=True, capture_output=True, text=True, timeout=TIMEOUT
        )

        if result.returncode != 0:
            return False, f"Compilation failed: {result.stderr}"

        return True, f"Compilation successful: {output_path}"

    def run(
        self,
        code: str,
        language: str,
        args: Optional[List[str]] = None,
        filename: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Run code in the specified language with optional arguments.
        Compiles the code first if needed.

        Args:
            code: The source code to run
            language: The programming language
            args: Optional list of command-line arguments
            filename: Optional filename to use

        Returns:
            Tuple of (success, output/error message)
        """
        language = language.lower()
        config = self.get_language_config(language)

        if not config:
            return False, f"No configuration found for language: {language}"

        # Sanitize code to remove non-printable characters
        code = sanitize_code(code, language)

        # Get file extension for this language
        extension = config.get("extension", language)

        # Create a temporary file with the code
        if filename:
            temp_dir = tempfile.mkdtemp()
            source_path = os.path.join(temp_dir, filename)
        else:
            fd, source_path = tempfile.mkstemp(suffix=f".{extension}")
            os.close(fd)

        with open(source_path, "w") as f:
            f.write(code)

        # Compile if needed
        if config.get("compile"):
            success, message = self.compile(code, language, filename)
            if not success:
                return False, message

        # Get the run command
        run_cmd = config.get("run")
        if not run_cmd:
            return False, f"No run command specified for language: {language}"

        logger.info(f"Original run_cmd: {run_cmd}, type: {type(run_cmd)}")

        # Fix for when run_cmd is a dictionary instead of a string
        if isinstance(run_cmd, dict):
            # For compiled languages, if the run command is a dictionary,
            # we assume it's {'output': None} and we use the compiled output path
            output_path = source_path.rsplit(".", 1)[0]
            run_cmd = output_path
            logger.info(f"Fixed run_cmd to use output path: {run_cmd}")
        else:
            # Replace placeholders in the run command
            run_cmd = run_cmd.replace("{source}", source_path)
            output_path = source_path.rsplit(".", 1)[0]
            run_cmd = run_cmd.replace("{output}", output_path)

        # Add command line arguments if provided
        if args:
            # Log the arguments for debugging
            logger.info(f"Arguments before processing: {args}")
            string_args = [str(arg) for arg in args]
            args_str = " ".join(string_args)
            run_cmd = f"{run_cmd} {args_str}"

        # Log the final command
        logger.info(f"Running command: {run_cmd}")

        # Run the command
        result = subprocess.run(
            run_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )

        output = result.stdout
        if result.returncode != 0:
            return False, f"Execution failed: {result.stderr}"

        return True, output

    def test(
        self, code: str, language: str, test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run test cases against the provided code.

        Args:
            code: The source code to test
            language: The programming language
            test_cases: List of test cases, each containing 'input' and 'expected' keys

        Returns:
            List of test results with status and output
        """
        results = []

        for i, test_case in enumerate(test_cases):
            test_input = test_case.get("input", [])
            expected = test_case.get("expected", "")

            # Run the code with the test input
            success, output = self.run(code, language, args=test_input)

            # Compare the output with the expected result
            passed = success and output.strip() == expected.strip()

            results.append(
                {
                    "test_case": i + 1,
                    "input": test_input,
                    "expected": expected,
                    "output": output,
                    "passed": passed,
                }
            )

        return results
