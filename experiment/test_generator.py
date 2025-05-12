"""
Test case generator for code translation experiments.
"""

import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

from utils.tools import LanguageTools, sanitize_code

logger = logging.getLogger("experiment.test_generator")


def generate_llm_test_inputs(
    task_description: str,
    source_code: str,
    language: str,
    num_test_cases: int = 5,
    llm_api_url: str = "http://localhost:1234/v1/chat/completions",
) -> List[List[str]]:
    """
    Generate test inputs using an LLM based on the problem description and source code.

    Args:
        task_description: Description of the programming task
        source_code: Source code implementation of the task
        language: Programming language of the implementation
        num_test_cases: Number of test cases to generate
        llm_api_url: URL for the LLM API server

    Returns:
        List of test inputs (each input is a list of strings)
    """
    prompt = f"""
You are helping generate test cases for a programming problem. Given a problem
description and a {language} implementation, suggest {num_test_cases} different
inputs to test the program thoroughly.\n

PROBLEM DESCRIPTION:
{task_description}

{language} IMPLEMENTATION:
```{language}
{source_code}
```

CAREFULLY ANALYZE IF THIS CODE REQUIRES ANY INPUTS:
- If the code simply performs a calculation or operation on FIXED values
(like formatting a constant value or displaying preset text), return an
empty list with a special flag: [["__NO_INPUT_NEEDED__"]]
- If the code expects command-line arguments or other inputs, provide appropriate
test inputs as shown below.

For each test case that needs inputs, provide the inputs that should be passed to the
program as command-line arguments. Return a list of test cases in the format:

[
  ["input1", "input2",...],  # Test case 1
  ["input1"],                # Test case 2
  ...
]

Your test cases should:
1. Cover edge cases and normal usage patterns
2. Include small, medium, and large inputs where appropriate
3. Test boundary conditions
4. Be valid for this specific program based on its implementation
5. Be realistic and diverse
6. Ensure that all inputs are interpretable by all languages

Return only the JSON array of test inputs, no explanations.
"""

    try:
        # Prepare the request for the LLM API
        payload = {
            "model": "any",  # The API will use whatever model it has
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI that generates test cases for code."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,  # Low temperature for more deterministic output
        }

        # Make the API call
        response = requests.post(
            llm_api_url, json=payload, headers={"Content-Type": "application/json"}
        )

        # Check if request was successful
        response.raise_for_status()

        # Parse the response
        result = response.json()
        output = result["choices"][0]["message"]["content"].strip()

        # Try to extract JSON array
        try:
            # Strip any markdown code block formatting
            if output.startswith("```") and output.endswith("```"):
                output = "\n".join(output.split("\n")[1:-1])

            test_inputs = json.loads(output)

            # Validate the result is a list of lists
            if not isinstance(test_inputs, list):
                raise ValueError("Expected a list of test inputs")

            # Make sure each element is a list
            test_inputs = [
                inputs if isinstance(inputs, list) else [inputs]
                for inputs in test_inputs
            ]

            return test_inputs

        except json.JSONDecodeError:
            logger.warning("Invalid JSON in LLM response, using fallback test inputs")
            # Return a fallback list
            return [["5"], ["0"], ["10"], ["20"], [str(random.randint(1, 100))]][
                :num_test_cases
            ]

    except Exception as e:
        logger.error(f"Error generating test inputs with LLM: {str(e)}")
        # Return a default set of inputs
        return [["5"], ["0"], ["10"], ["20"], [str(random.randint(1, 100))]][
            :num_test_cases
        ]


def generate_test_cases(
    problem: Dict[str, Any],
    num_test_cases: int = 5,
    output_path: Optional[str] = None,
    source_language: str = "C",
    llm_api_url: str = "http://localhost:1234/v1/chat/completions",
) -> List[Dict[str, Any]]:
    """
    Generate test cases for a programming problem using an existing implementation.

    Args:
        problem: Problem dictionary with task description and implementations
        num_test_cases: Number of test cases to generate
        output_path: Optional path to save test cases to
        source_language: Language to use for generating test cases (defaults to C)
        llm_api_url: URL for the LLM API server

    Returns:
        List of test cases with input and expected output
    """
    task_name = problem["task_name"]
    task_description = problem["task_description"]
    implementations = problem["implementations"]

    logger.info(f"Generating {num_test_cases} test cases for problem: {task_name}")

    # Check if this might be a formatting problem based on the task name
    is_formatting_problem = any(
        keyword in task_name.lower()
        for keyword in ["format", "formatting", "output", "display"]
    )

    if is_formatting_problem:
        logger.info(
            (
                f"This appears to be a formatting problem, checking if input needed: "
                f"{task_name}"
            )
        )

    # Check if source language implementation exists
    if source_language not in implementations:
        # Fall back to Python if specified source language not available
        if "Python" in implementations:
            source_language = "Python"
            logger.info("Specified source language not found, falling back to Python")
        else:
            # Try to find another language that's easy to work with
            preferred_languages = ["Javascript", "Ruby", "Java", "C#", "C++", "C"]
            for lang in preferred_languages:
                if lang in implementations:
                    source_language = lang
                    logger.info(f"Using {lang} implementation for test case generation")
                    break
            else:
                # If no preferred language found, use the first available
                source_language = next(iter(implementations))
                logger.info(
                    f"Using {source_language} implementation for test case generation"
                )

    # Get source code and sanitize it
    source_code = implementations[source_language]
    if isinstance(source_code, str):  # Add this check
        source_code = sanitize_code(source_code, source_language)
    else:
        logger.error(
            f"Invalid source code format for {source_language}: {type(source_code)}"
        )
        return []

    logger.info(f"Using {source_language} implementation for test case generation")

    # Initialize language tools to get the correct extension
    language_tools = LanguageTools()
    lang_config = language_tools.get_language_config(source_language.lower())

    if not lang_config:
        logger.warning(f"No configuration found for language: {source_language}")
        extension = source_language.lower()
    else:
        extension = lang_config.get("extension", source_language.lower())

    logger.info(f"Using file extension '{extension}' for {source_language}")

    # For formatting problems
    if is_formatting_problem:
        try:
            # Create a temporary file for the source code
            with tempfile.NamedTemporaryFile(
                suffix=f".{extension}", mode="w", delete=False
            ) as tmp:
                tmp.write(source_code)
                source_file = tmp.name

            # Try running without any input
            success, output = language_tools.run(
                source_code,
                source_language,
                args=[],  # No arguments
                filename=Path(source_file).name,
            )

            if success and output.strip():
                logger.info(
                    f"Successfully ran formatting problem without input: {task_name}"
                )

                # Create test case for formatting output
                test_cases = [
                    {
                        "input": [],
                        "expected": output.strip(),
                        "no_input_needed": True,
                        "formatting_problem": True,
                    }
                ]

                # Save test cases if output path is specified
                if output_path:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w") as f:
                        json.dump(test_cases, f, indent=2)
                    logger.info(f"Saved test cases to {output_path}")

                # Clean up temporary file
                if os.path.exists(source_file):
                    os.unlink(source_file)

                return test_cases
        except Exception as e:
            logger.warning(
                f"Error trying to run formatting problem without input: {str(e)}"
            )
            # Continue with regular test case generation

    # Create a temporary file for the source code with the correct extension
    with tempfile.NamedTemporaryFile(
        suffix=f".{extension}", mode="w", delete=False
    ) as tmp:
        tmp.write(source_code)
        source_file = tmp.name

    try:
        # Generate test cases
        test_cases = []

        # Generate test inputs using LLM
        try:
            test_inputs = generate_llm_test_inputs(
                task_description=task_description,
                source_code=source_code,
                language=source_language,
                num_test_cases=num_test_cases,
                llm_api_url=llm_api_url,
            )

            # Check if this is a function that doesn't need input
            if len(test_inputs) == 1 and test_inputs[0] == ["__NO_INPUT_NEEDED__"]:
                logger.info(
                    f"Detected function that doesn't need input for {task_name}"
                )

                # Run the implementation with no inputs
                success, output = language_tools.run(
                    source_code,
                    source_language,
                    args=[],  # No arguments needed
                    filename=Path(source_file).name,
                )

                if success:
                    # Create a special test case with no input
                    test_case = {
                        "input": [],
                        "expected": output.strip(),
                        "no_input_needed": True,
                    }
                    test_cases.append(test_case)
                    logger.info(
                        f"Generated fixed output test case: -> {output.strip()}"
                    )
                else:
                    logger.warning(f"Failed to run without input: {output}")

                # For functions without input
                if len(test_cases) > 0:
                    # Save test cases if output path is specified
                    if output_path:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        with open(output_path, "w") as f:
                            json.dump(test_cases, f, indent=2)
                        logger.info(f"Saved test cases to {output_path}")
                    return test_cases

        except Exception as e:
            logger.error(f"Error using LLM for test generation: {str(e)}")
            # Fall back to default inputs if LLM fails
            test_inputs = [
                ["5"],  # Small input
                ["0"],  # Zero/empty input
                ["10"],  # Medium input
                ["20"],  # Larger input
                [str(random.randint(1, 100))],  # Random input
            ][:num_test_cases]

        # Run each test input through the implementation
        for i, input_args in tqdm(
            enumerate(test_inputs),
            total=len(test_inputs),
            desc="Testing inputs",
            unit="test",
            leave=False,
        ):
            logger.info(f"Running test input {i + 1} ({source_language}): {input_args}")

            # Run the implementation with the input
            success, output = language_tools.run(
                source_code,
                source_language,
                args=input_args,
                filename=Path(source_file).name,
            )

            if success:
                # Create test case with the input and resulting output
                test_case = {"input": input_args, "expected": output.strip()}
                test_cases.append(test_case)
                logger.info(
                    f"Generated test case {len(test_cases)}:"
                    f" {input_args} -> {output.strip()}"
                )
            else:
                logger.warning(f"Failed to run with input {input_args}: {output}")

        # If we didn't get enough test cases, add some more with different inputs
        additional_attempts = 0
        while len(test_cases) < num_test_cases and additional_attempts < 10:
            additional_attempts += 1
            input_args = [str(random.randint(1, 1000))]

            success, output = language_tools.run(
                source_code,
                source_language,
                args=input_args,
                filename=Path(source_file).name,
            )

            if success:
                # Check if this is a duplicate test case
                if not any(tc["input"] == input_args for tc in test_cases):
                    test_case = {"input": input_args, "expected": output.strip()}
                    test_cases.append(test_case)
                    logger.info(
                        (
                            f"Generated additional test case {len(test_cases)}:"
                            f" {input_args} -> {output.strip()}"
                        )
                    )

        logger.info(f"Generated {len(test_cases)} test cases for {task_name}")

        # Save test cases if output path is specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(test_cases, f, indent=2)
            logger.info(f"Saved test cases to {output_path}")

        return test_cases

    except Exception as e:
        logger.error(f"Error generating test cases: {str(e)}")
        # Return an empty list in case of error
        return []
    finally:
        # Clean up temporary file
        if os.path.exists(source_file):
            os.unlink(source_file)


def load_test_cases(test_cases_path: str) -> List[Dict[str, Any]]:
    """
    Load test cases from a JSON file.

    Args:
        test_cases_path: Path to the test cases JSON file

    Returns:
        List of test cases
    """
    with open(test_cases_path, "r") as f:
        return json.load(f)
