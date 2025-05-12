"""
Evaluation module for code translation experiments.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import polars as pl
from tqdm import tqdm

logger = logging.getLogger("experiment.evaluator")


def evaluate_translation(
    source_file: Union[str, Path],
    language: str,
    test_cases: List[Dict[str, Any]],
    language_tools: Any,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a code translation by testing its compilation and execution.

    Args:
        source_file: Path to the source code file
        language: Programming language
        test_cases: List of test cases to run
        language_tools: LanguageTools instance
        output_path: Optional path to save evaluation results

    Returns:
        Dictionary with evaluation results
    """
    source_file = Path(source_file)
    language = language.lower()

    logger.info(f"Evaluating {language} translation: {source_file}")

    # Read the source code
    with open(source_file, "r") as f:
        code = f.read()

    # Initialize results
    results = {
        "language": language,
        "source_file": str(source_file),
        "compiles": None,
        "compile_output": None,
        "test_results": [],
    }

    # First, check if the language needs compilation
    lang_config = language_tools.get_language_config(language)
    needs_compilation = "compile" in lang_config and lang_config["compile"]

    if needs_compilation:
        # Try to compile the code
        success, output = language_tools.compile(
            code, language, filename=source_file.name
        )

        results["compiles"] = success
        results["compile_output"] = output

        if not success:
            logger.warning(f"Compilation failed for {language} file: {source_file}")

            # Save evaluation results if output path is specified
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

            return results
    else:
        # If language doesn't need compilation, mark as compiled
        results["compiles"] = True
        results["compile_output"] = "Language does not require compilation"

    # Run test cases
    for i, test_case in tqdm(
        enumerate(test_cases),
        total=len(test_cases),
        desc=f"Running tests ({language})",
        unit="test",
        leave=False,
    ):
        try:
            # Check if this is a function that doesn't need input
            is_no_input_function = test_case.get("no_input_needed", False)

            # Get input data and ensure it's in the correct format
            test_input = test_case.get("input", [])

            # If test_input is a dictionary, convert it to a list of args
            if isinstance(test_input, dict):
                # Convert dict to a list of command line args (e.g. "--key=value")
                processed_args = []
                for key, value in test_input.items():
                    processed_args.append(f"--{key}={value}")
                test_input = processed_args
            elif not isinstance(test_input, list):
                # If it's not a list (like a single string or number), wrap it in a list
                test_input = [str(test_input)]

            # Run code with test input (or without input for fixed-output functions)
            success, output = language_tools.run(
                code,
                language,
                args=test_input,
                filename=source_file.name,
            )

            # Compare with expected output
            expected = test_case.get("expected", "").strip()
            actual = output.strip()
            passed = success and actual == expected

            # If this is a fixed output function, we might need more flexible comparison
            # since the exact formatting or whitespace might differ
            if is_no_input_function and not passed:
                # Try more flexible comparison by removing all whitespace
                stripped_expected = re.sub(r"\s+", "", expected)
                stripped_actual = re.sub(r"\s+", "", actual)
                if stripped_expected == stripped_actual:
                    passed = True
                    logger.info("Test passed with whitespace-insensitive comparison")

            # Store test result
            test_result = {
                "test_case": i + 1,
                "input": test_input,  # Use the processed input
                "original_input": test_case.get(
                    "input", []
                ),  # Store original input for reference
                "expected": expected,
                "actual": actual,
                "success": success,
                "passed": passed,
                "error": None if success else output,
                "no_input_needed": is_no_input_function,
            }

            results["test_results"].append(test_result)

            if passed:
                logger.info(
                    f"Test case {i + 1} passed for {language} file: {source_file}"
                )
            else:
                logger.warning(
                    f"Test case {i + 1} failed for {language} file: {source_file}"
                )
                if not success:
                    logger.warning(f"Error: {output}")
                else:
                    logger.warning(f"Expected: '{expected}', Got: '{actual}'")

        except Exception as e:
            # Handle unexpected errors
            logger.error(
                f"Error running test case {i + 1} for {language} file: {str(e)}"
            )

            test_result = {
                "test_case": i + 1,
                "input": test_case.get("input", []),
                "expected": test_case.get("expected", "").strip(),
                "actual": None,
                "success": False,
                "passed": False,
                "error": str(e),
                "no_input_needed": test_case.get("no_input_needed", False),
            }

            results["test_results"].append(test_result)

    # Calculate pass rate
    if results["test_results"]:
        pass_rate = sum(1 for tc in results["test_results"] if tc["passed"]) / len(
            results["test_results"]
        )
        results["pass_rate"] = pass_rate
        logger.info(f"Pass rate for {language} file: {pass_rate:.2%}")
    else:
        results["pass_rate"] = 0.0

    # Save evaluation results if output path is specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def generate_experiment_report(
    experiment_state: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Generate a summary report of the experiment.

    Args:
        experiment_state: Experiment state dictionary
        output_path: Optional path to save the report

    Returns:
        Report as a string
    """
    logger.info("Generating experiment report")

    # Extract experiment information
    experiment_name = experiment_state.get("name", "Unknown")
    started_at = experiment_state.get("started_at", "Unknown")
    problems = experiment_state.get("problems", {})
    metrics = experiment_state.get("metrics", {})

    compilation_success = metrics.get("compilation_success", {})
    test_case_success = metrics.get("test_case_success", {})

    # Generate report sections
    report = [
        f"# Experiment Report: {experiment_name}",
        f"*Generated at: {started_at}*\n",
        "## Summary",
        f"- **Number of problems:** {len(problems)}",
        "- **Number of languages:** ",
        f"{
            len(
                test_case_success[next(iter(test_case_success))]
                if test_case_success
                else {}
            )
        }",
    ]

    # Add problem list
    report.append("\n## Problems")
    for problem_name, problem_info in problems.items():
        report.append(f"### {problem_name}")
        report.append(f"*{problem_info.get('task_description', 'No description')}*\n")

        report.append("**Implementations:**")
        report.append(", ".join(problem_info.get("implementations", [])))

        if "pseudocode" in problem_info:
            report.append("\n**Source Language for Pseudocode:**")
            report.append(problem_info["pseudocode"].get("source_language", "Unknown"))

        report.append("")

    # Calculate overall metrics
    if compilation_success and test_case_success:
        # Compile success rate by language
        lang_compilation_rates = {}
        for _, langs in compilation_success.items():
            for lang, success in langs.items():
                if lang not in lang_compilation_rates:
                    lang_compilation_rates[lang] = []
                lang_compilation_rates[lang].append(int(success))

        overall_compilation = {
            lang: (sum(rates) / len(rates) if rates else 0)
            for lang, rates in lang_compilation_rates.items()
        }

        # Test case success rate by language
        lang_test_rates = {}
        for _, langs in test_case_success.items():
            for lang, rate in langs.items():
                if lang not in lang_test_rates:
                    lang_test_rates[lang] = []
                lang_test_rates[lang].append(float(rate))

        overall_test_success = {
            lang: (sum(rates) / len(rates) if rates else 0)
            for lang, rates in lang_test_rates.items()
        }

        # Add overall metrics section
        report.append("## Overall Results\n")

        # Create metrics table
        report.append("### Performance by Language\n")
        report.append("| Language | Compilation Success | Test Case Success |")
        report.append("|----------|-------------------|------------------|")

        for lang in sorted(
            set(list(overall_compilation.keys()) + list(overall_test_success.keys()))
        ):
            comp_rate = overall_compilation.get(lang, 0) * 100
            test_rate = overall_test_success.get(lang, 0) * 100
            report.append(f"| {lang} | {comp_rate:.1f}% | {test_rate:.1f}% |")

        # Generate charts
        if output_path:
            # Ensure the directory exists
            report_dir = Path(output_path).parent
            os.makedirs(report_dir, exist_ok=True)
            charts_dir = report_dir / "charts"
            os.makedirs(charts_dir, exist_ok=True)

            # Create DataFrame for visualization
            data = []
            for lang in sorted(
                set(
                    list(overall_compilation.keys()) + list(overall_test_success.keys())
                )
            ):
                data.append(
                    {
                        "Language": lang,
                        "Compilation Success": overall_compilation.get(lang, 0) * 100,
                        "Test Case Success": overall_test_success.get(lang, 0) * 100,
                    }
                )

            df = pl.DataFrame(data)

            # Compilation success chart
            plt.figure(figsize=(10, 6))
            plt.bar(df["Language"], df["Compilation Success"], color="blue")
            plt.title("Compilation Success Rate by Language")
            plt.xlabel("Language")
            plt.ylabel("Success Rate (%)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            comp_chart_path = charts_dir / "compilation_success.png"
            plt.savefig(comp_chart_path)
            plt.close()

            # Test case success chart
            plt.figure(figsize=(10, 6))
            plt.bar(df["Language"], df["Test Case Success"], color="green")
            plt.title("Test Case Success Rate by Language")
            plt.xlabel("Language")
            plt.ylabel("Success Rate (%)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            test_chart_path = charts_dir / "test_case_success.png"
            plt.savefig(test_chart_path)
            plt.close()

            # Combined chart
            plt.figure(figsize=(12, 6))
            width = 0.35
            x = range(len(df))
            plt.bar(
                [i - width / 2 for i in x],
                df["Compilation Success"],
                width,
                label="Compilation",
                color="blue",
            )
            plt.bar(
                [i + width / 2 for i in x],
                df["Test Case Success"],
                width,
                label="Test Cases",
                color="green",
            )
            plt.title("Success Rates by Language")
            plt.xlabel("Language")
            plt.ylabel("Success Rate (%)")
            plt.xticks(x, df["Language"], rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()
            combined_chart_path = charts_dir / "combined_success.png"
            plt.savefig(combined_chart_path)
            plt.close()

            # Add charts to the report
            report.append("\n### Charts\n")
            report.append(
                "![Compilation Success Rate](charts/compilation_success.png)\n"
            )
            report.append("![Test Case Success Rate](charts/test_case_success.png)\n")
            report.append("![Combined Success Rates](charts/combined_success.png)\n")

    # Add detailed results by problem
    report.append("## Detailed Results by Problem\n")

    for problem_name in problems:
        report.append(f"### {problem_name}\n")

        if problem_name in compilation_success:
            # Compilation results
            report.append("#### Compilation Results\n")
            report.append("| Language | Compilation Success |")
            report.append("|----------|-------------------|")

            for lang, success in sorted(compilation_success[problem_name].items()):
                status = "✅ Success" if success else "❌ Failed"
                report.append(f"| {lang} | {status} |")

            report.append("")

        if problem_name in test_case_success:
            # Test case results
            report.append("#### Test Case Results\n")
            report.append("| Language | Test Case Success Rate |")
            report.append("|----------|-----------------------|")

            for lang, success_rate in sorted(test_case_success[problem_name].items()):
                report.append(f"| {lang} | {success_rate * 100:.1f}% |")

            report.append("")

    # Combine report sections into a single string
    report_text = "\n".join(report)

    # Save report if output path is specified
    if output_path:
        with open(output_path, "w") as f:
            f.write(report_text)
        logger.info(f"Saved experiment report to {output_path}")

    return report_text
