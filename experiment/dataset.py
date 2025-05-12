"""
Dataset handling module for code translation experiments.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from utils.data import generate_test_set as get_filtered_dataset
from utils.definitions import BLACKLISTED_PROBLEMS

logger = logging.getLogger("experiment.dataset")


def sanitize_filename(name: str) -> str:
    """
    Create a filesystem-safe filename from any string.

    Args:
        name: Original string that may contain invalid characters

    Returns:
        Sanitized filename without problematic characters
    """
    # Remove or replace characters that could cause issues in filenames
    # Replace spaces and special characters with underscores
    # sanitized = re.sub(r"[^\w\-\.]", "_", name)
    # Convert to lowercase for consistency
    sanitized = name.lower()
    # Ensure we don't have any path traversal
    sanitized = sanitized.replace("/", "_").replace("\\", "_")
    return sanitized


def generate_test_set(
    num_problems: int = 10,
    min_implementations: int = 5,
    output_dir: Optional[str] = None,
    languages: Optional[List[str]] = None,
    require_all_languages: bool = False,
    use_cleaned_code: bool = True,
    llm_api_url: str = "http://localhost:1234/v1/chat/completions",
    batch_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Generate a test set of problems from the Rosetta Code dataset.

    Args:
        num_problems: Number of problems to include in the test set
        min_implementations: Minimum number of language implementations
            required for a problem
        output_dir: Directory to save the problems to
        languages: List of languages to focus on
        require_all_languages: If True, only include problems that have implementations
            in all selected languages
        use_cleaned_code: Whether to use LLM-cleaned code that removes examples and
            unnecessary content
        llm_api_url: URL for the LLM API endpoint
        batch_size: Number of samples to process in each batch for LLM cleaning

    Returns:
        List of problems with their implementations
    """
    logger.info(
        (
            f"Generating test set with {num_problems} problems"
            f" (min implementations: {min_implementations})"
        )
    )

    # Get the Rosetta Code dataset with filtered languages and cleaned code
    df = get_filtered_dataset(
        languages=languages,
        use_cleaned_code=use_cleaned_code,
        llm_api_url=llm_api_url,
        batch_size=batch_size,
    )

    # Filter out blacklisted problems
    df = df.filter(~pl.col("task_name").str.to_lowercase().is_in(BLACKLISTED_PROBLEMS))

    # Group by task name and count implementations
    task_counts = df.group_by("task_name").agg(
        pl.count("language_name").alias("implementation_count"),
        pl.col("task_description").first().alias("task_description"),
    )

    # Filter tasks by minimum number of implementations
    qualified_tasks = task_counts.filter(
        pl.col("implementation_count") >= min_implementations
    )
    logger.info(
        (
            f"Found {qualified_tasks.height} tasks with at least"
            f" {min_implementations} implementations"
        )
    )

    # If we need to ensure all problems have all required languages
    if require_all_languages and languages:
        # Further filter tasks to only those with all required languages
        qualified_task_names = []

        # Check each qualified task to see if it has all required languages
        for task_name in qualified_tasks["task_name"]:
            # Get all implementations for this task
            task_implementations = df.filter(pl.col("task_name") == task_name)

            # Check if all required languages are present
            available_languages = set(task_implementations["language_name"].to_list())
            required_languages = set(
                lang.title() for lang in languages
            )  # Ensure proper case

            if required_languages.issubset(available_languages):
                qualified_task_names.append(task_name)

        # Filter the qualified tasks to only those with all required languages
        qualified_tasks = qualified_tasks.filter(
            pl.col("task_name").is_in(qualified_task_names)
        )

        logger.info(
            (
                f"Found {qualified_tasks.height} tasks with "
                f"implementations in all required languages"
            )
        )

        # If we don't have enough qualified tasks, warn the user
        if qualified_tasks.height < num_problems:
            logger.warning(
                (
                    f"Only {qualified_tasks.height} tasks have "
                    f"implementations in all required languages. "
                    f"Requested {num_problems} problems."
                )
            )

    # Randomly select problems
    if num_problems == -1:
        selected_task_indices = range(qualified_tasks.height)
    else:
        selected_task_indices = random.sample(
            range(qualified_tasks.height), min(num_problems, qualified_tasks.height)
        )
    selected_tasks = qualified_tasks[selected_task_indices, :]
    selected_task_names = selected_tasks["task_name"].to_list()

    logger.info(f"Selected {len(selected_task_names)} tasks for the test set")

    # Get implementations for the selected tasks
    implementations = df.filter(pl.col("task_name").is_in(selected_task_names))

    # Convert to list of problem dictionaries
    problems = []
    for task_name in selected_task_names:
        task_impl = implementations.filter(pl.col("task_name") == task_name)
        task_desc = task_impl["task_description"][0]

        problem = {
            "task_name": task_name,
            "task_description": task_desc,
            "implementations": {},
        }

        # Add implementations
        for row in task_impl.rows(named=True):
            problem["implementations"][row["language_name"]] = row["code"]

        problems.append(problem)

    # Save problems to output directory if specified
    if output_dir:
        # Create the main output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save a copy of all problems in a single file
        all_problems_file = output_path / "all_problems.json"
        with open(all_problems_file, "w") as f:
            json.dump(problems, f, indent=2)
            logger.info(f"Saved all problems to {all_problems_file}")

        for problem in problems:
            # Sanitize the task name for file system use
            safe_task_name = sanitize_filename(problem["task_name"])

            # Create the problem-specific directory
            problem_dir = output_path / safe_task_name
            problem_dir.mkdir(parents=True, exist_ok=True)

            # Save the problem json in its directory
            problem_file = problem_dir / "problem.json"
            with open(problem_file, "w") as f:
                json.dump(problem, f, indent=2)
                logger.info(f"Saved problem to {problem_file}")

            # Save individual language implementations
            for lang, code in problem["implementations"].items():
                # Create a safe filename for the language
                safe_lang_name = sanitize_filename(lang)
                lang_file = problem_dir / f"{safe_lang_name}.txt"

                # Write the implementation file
                with open(lang_file, "w") as f:
                    f.write(code)

    logger.info(f"Generated test set with {len(problems)} problems")
    return problems


def load_problem(problem_path: str) -> Dict[str, Any]:
    """
    Load a problem from a JSON file.

    Args:
        problem_path: Path to the problem JSON file

    Returns:
        Problem dictionary
    """
    with open(problem_path, "r") as f:
        return json.load(f)


def get_language_implementations(
    problem: Dict[str, Any], languages: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Get implementations of a problem in specific languages.

    Args:
        problem: Problem dictionary
        languages: Optional list of languages to filter by

    Returns:
        Dictionary mapping language names to code implementations
    """
    if languages is None:
        return problem["implementations"]

    # Filter implementations to only include specified languages
    lowercase_languages = [lang.lower() for lang in languages]
    return {
        lang: code
        for lang, code in problem["implementations"].items()
        if lang.lower() in lowercase_languages
    }
