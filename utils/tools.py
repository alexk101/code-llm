import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


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

        try:
            # Run the compile command
            result = subprocess.run(
                compile_cmd, shell=True, capture_output=True, text=True
            )

            if result.returncode != 0:
                return False, f"Compilation failed: {result.stderr}"

            return True, f"Compilation successful: {output_path}"

        except Exception as e:
            return False, f"Error during compilation: {str(e)}"

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

        # Replace placeholders in the run command
        run_cmd = run_cmd.replace("{source}", source_path)
        output_path = source_path.rsplit(".", 1)[0]
        run_cmd = run_cmd.replace("{output}", output_path)

        # Add command line arguments if provided
        if args:
            args_str = " ".join(args)
            run_cmd = f"{run_cmd} {args_str}"

        try:
            # Run the command
            result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)

            output = result.stdout
            if result.returncode != 0:
                return False, f"Execution failed: {result.stderr}"

            return True, output

        except Exception as e:
            return False, f"Error during execution: {str(e)}"

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
