"""
Core experiment module for code translation experiments.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from experiment.dataset import sanitize_filename
from utils.tools import LanguageTools

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("experiment")


class Experiment:
    """
    Main experiment class for code translation and evaluation.

    This class orchestrates:
    1. Dataset collection from Rosetta Code
    2. Test case generation for programming problems
    3. Pseudocode generation from source language
    4. Translation from pseudocode to target languages
    5. Testing translations against test cases
    6. Recording metrics (compilation success, test case pass rates)
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "experiment_results",
        languages_config: str = "language_tools.yaml",
        use_top_n_languages: int = 20,
        llm_api_url: str = "http://localhost:1234/v1/chat/completions",
    ):
        """
        Initialize experiment.

        Args:
            experiment_name: Name of this experiment run
            output_dir: Directory to store experiment results
            languages_config: Path to language tools configuration file
            use_top_n_languages: Number of top languages to include from TIOBE
            llm_api_url: URL for the LLM API server
        """
        self.experiment_name = experiment_name
        self._output_dir = Path(output_dir) / experiment_name
        self.languages_config = languages_config
        self.use_top_n_languages = use_top_n_languages
        self.llm_api_url = llm_api_url

        # Create output directory
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize language tools
        self.language_tools = LanguageTools(languages_config)

        # Load language set
        self.languages = self._get_top_languages(use_top_n_languages)

        # Track experiment progress
        self.experiment_state = {
            "name": experiment_name,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "problems": {},
            "translations": {},
            "metrics": {"compilation_success": {}, "test_case_success": {}},
        }

        # Save initial state
        self.save_state()

        logger.info(
            (
                f"Initialized experiment '{experiment_name}' with top"
                f" {use_top_n_languages} languages"
            )
        )

    @property
    def output_dir(self) -> Path:
        """Get the experiment output directory."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: str) -> None:
        """
        Set the experiment output directory.

        This also updates the experiment state and creates the new directory.

        Args:
            value: The new output directory path
        """
        # Convert to Path if it's a string
        if isinstance(value, str):
            value = Path(value)

        # Update the directory
        self._output_dir = value

        # Create the directory if it doesn't exist
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Update experiment state
        self.experiment_state["name"] = self.experiment_name

        logger.info(f"Output directory changed to: {self._output_dir}")

        # Save updated state to the new location
        self.save_state()

    def _get_top_languages(self, top_n: int) -> List[str]:
        """
        Get the top N programming languages from TIOBE index.

        Args:
            top_n: Number of top languages to include

        Returns:
            List of language names
        """
        from utils.data import get_language_info
        from utils.validate_tools import verify_language_tools

        # Get top languages from language info
        language_df = get_language_info()
        top_languages = language_df["Programming Language"].head(top_n).to_list()

        # Verify language tools using validate_tools
        missing_languages, present_languages = verify_language_tools()

        if missing_languages:
            logger.warning(
                (
                    "Some languages are not supported",
                    f"by language tools: {missing_languages}",
                )
            )

        # Use only the verified present languages that are in our top_n list
        supported_languages = [
            lang for lang in present_languages if lang in top_languages
        ]

        logger.info(
            f"Using {len(supported_languages)} supported languages from top {top_n}"
        )

        return supported_languages

    def save_state(self) -> None:
        """Save current experiment state to disk."""
        state_path = self.output_dir / "experiment_state.json"
        with open(state_path, "w") as f:
            json.dump(self.experiment_state, f, indent=2)
        logger.info(f"Saved experiment state to {state_path}")

    def generate_test_set(
        self,
        num_problems: int = 10,
        min_implementations: int = 5,
        problems_output_dir: Optional[str] = None,
        require_all_languages: bool = False,
        use_cleaned_code: bool = True,
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate a test set of problems from Rosetta Code dataset.

        Args:
            num_problems: Number of problems to include in test set
            min_implementations: Minimum number of implementations required
                for a problem
            problems_output_dir: Optional directory to save problems to
            require_all_languages: If True, only include problems with implementations
                in all target languages
            use_cleaned_code: Whether to use LLM-cleaned code that removes examples
            batch_size: Number of samples to process in each batch for LLM cleaning

        Returns:
            List of problems with their implementations
        """
        from experiment.dataset import generate_test_set as gen_test_set

        # Default problems directory is inside experiment output directory
        if problems_output_dir is None:
            problems_output_dir = self.output_dir / "problems"
        os.makedirs(problems_output_dir, exist_ok=True)

        # Generate test set
        problems = gen_test_set(
            num_problems=num_problems,
            min_implementations=min_implementations,
            output_dir=problems_output_dir,
            languages=self.languages,
            require_all_languages=require_all_languages,
            use_cleaned_code=use_cleaned_code,
            llm_api_url=self.llm_api_url,
            batch_size=batch_size,
        )

        # Update experiment state
        self.experiment_state["problems"] = {
            p["task_name"]: {
                "task_description": p["task_description"],
                "implementations": list(p["implementations"].keys()),
            }
            for p in problems
        }
        self.save_state()

        return problems

    def generate_test_cases(
        self,
        problems: List[Dict[str, Any]],
        num_test_cases: int = 5,
        test_cases_output_dir: Optional[str] = None,
        source_language: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate test cases for each problem.

        Args:
            problems: List of problems to generate test cases for
            num_test_cases: Number of test cases to generate per problem
            test_cases_output_dir: Optional directory to save test cases to
            source_language: Optional language to prefer for test case generation

        Returns:
            Dictionary mapping problem names to test cases
        """
        from experiment.dataset import generate_test_set as gen_test_set
        from experiment.test_generator import generate_test_cases

        # Default test cases directory is inside experiment output directory
        if test_cases_output_dir is None:
            test_cases_output_dir = self.output_dir / "test_cases"
        os.makedirs(test_cases_output_dir, exist_ok=True)

        # Generate test cases for each problem
        all_test_cases = {}

        # Keep track of which problems we've processed and which have failed
        processed_problems = set()
        failed_problems = set()
        remaining_problems = list(problems)

        # Continue until we've processed all problems or have exhausted our options
        while remaining_problems:
            problem = remaining_problems.pop(0)
            problem_name = problem["task_name"]

            # Skip if we've already processed this problem
            if problem_name in processed_problems:
                continue

            processed_problems.add(problem_name)

            # Create a safe directory name for this problem
            safe_task_name = sanitize_filename(problem_name)

            # Create problem-specific directory
            problem_dir = Path(test_cases_output_dir) / safe_task_name
            problem_dir.mkdir(parents=True, exist_ok=True)

            # Generate test cases
            test_cases = generate_test_cases(
                problem=problem,
                num_test_cases=num_test_cases,
                output_path=problem_dir / "test_cases.json",
                source_language=source_language,
                llm_api_url=self.llm_api_url,
            )

            # If test case generation failed, try a different problem
            if not test_cases:
                logger.warning(
                    (f"Failed to generate test cases for problem: {problem_name}")
                )
                failed_problems.add(problem_name)

                # Remove the problem files and directories
                try:
                    # Remove test cases directory for this problem
                    problem_test_dir = Path(test_cases_output_dir) / safe_task_name
                    if problem_test_dir.exists():
                        shutil.rmtree(problem_test_dir)
                        logger.info(
                            (
                                f"Removed test case directory for failed problem: "
                                f"{problem_name}"
                            )
                        )

                    # Remove problem directory from problems directory
                    problems_dir = Path(test_cases_output_dir).parent / "problems"
                    problem_dir = problems_dir / safe_task_name
                    if problem_dir.exists():
                        shutil.rmtree(problem_dir)
                        logger.info(
                            (
                                f"Removed problem directory for failed problem: "
                                f"{problem_name}"
                            )
                        )

                        # Also update the all_problems.json file
                        all_problems_file = problems_dir / "all_problems.json"
                        if all_problems_file.exists():
                            try:
                                with open(all_problems_file, "r") as f:
                                    all_probs = json.load(f)

                                # Remove the failed problem
                                all_probs = [
                                    p
                                    for p in all_probs
                                    if p["task_name"] != problem_name
                                ]

                                # Write back the updated list
                                with open(all_problems_file, "w") as f:
                                    json.dump(all_probs, f, indent=2)
                                logger.info(
                                    (
                                        f"Updated all_problems.json by removing "
                                        f"{problem_name}"
                                    )
                                )
                            except Exception as e:
                                logger.error(
                                    (f"Error updating all_problems.json: {str(e)}")
                                )

                    # Remove from experiment state
                    if problem_name in self.experiment_state["problems"]:
                        del self.experiment_state["problems"][problem_name]
                        logger.info(
                            (
                                f"Removed failed problem from experiment state: "
                                f"{problem_name}"
                            )
                        )
                except Exception as e:
                    logger.error(
                        (
                            f"Error removing files for failed problem {problem_name}: "
                            f"{str(e)}"
                        )
                    )

                # Try to get a replacement problem if we don't have enough successes
                if len(all_test_cases) < len(problems) - len(failed_problems):
                    logger.info(
                        (
                            f"Attempting to find a replacement for failed problem: "
                            f"{problem_name}"
                        )
                    )

                    # Generate a new problem to replace the failed one
                    try:
                        # Get one additional problem with the same requirements
                        replacement_problems = gen_test_set(
                            num_problems=1,
                            min_implementations=5,  # Use a reasonable default
                            languages=self.languages,
                            llm_api_url=self.llm_api_url,
                        )

                        if replacement_problems:
                            new_problem = replacement_problems[0]
                            new_name = new_problem["task_name"]

                            # Make sure we don't have duplicates
                            if new_name not in processed_problems and new_name not in [
                                p["task_name"] for p in remaining_problems
                            ]:
                                logger.info(f"Found replacement problem: {new_name}")
                                remaining_problems.append(new_problem)

                                # Save the new problem to the problems directory
                                if test_cases_output_dir:
                                    problems_dir = (
                                        Path(test_cases_output_dir).parent / "problems"
                                    )
                                    if problems_dir.exists():
                                        # Sanitize the task name for file system use
                                        safe_new_name = sanitize_filename(new_name)

                                        # Create the problem-specific directory
                                        new_problem_dir = problems_dir / safe_new_name
                                        new_problem_dir.mkdir(
                                            parents=True, exist_ok=True
                                        )

                                        # Save the problem json in its directory
                                        problem_file = new_problem_dir / "problem.json"
                                        with open(problem_file, "w") as f:
                                            json.dump(new_problem, f, indent=2)
                                            logger.info("Saved replacement problem")
                    except Exception as e:
                        logger.error(f"Error finding replacement problem: {str(e)}")
                continue

            all_test_cases[problem_name] = test_cases

            # Update experiment state
            if problem_name in self.experiment_state["problems"]:
                self.experiment_state["problems"][problem_name]["test_cases"] = (
                    test_cases
                )

        self.save_state()

        if failed_problems:
            logger.warning(
                (
                    f"Failed to generate test cases for {len(failed_problems)} "
                    f"problems: {', '.join(failed_problems)}"
                )
            )

        logger.info(
            f"Successfully generated test cases for {len(all_test_cases)} problems"
        )
        return all_test_cases

    def generate_pseudocode(
        self,
        problems: List[Dict[str, Any]],
        source_language: str,
        pseudocode_output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate pseudocode for each problem from source language.

        Args:
            problems: List of problems to generate pseudocode for
            source_language: Source language to generate pseudocode from
            pseudocode_output_dir: Optional directory to save pseudocode to

        Returns:
            Dictionary mapping problem names to pseudocode
        """
        from experiment.translator import generate_pseudocode

        # Default pseudocode directory is inside experiment output directory
        if pseudocode_output_dir is None:
            pseudocode_output_dir = self.output_dir / "pseudocode"
        os.makedirs(pseudocode_output_dir, exist_ok=True)

        # Generate pseudocode for each problem
        all_pseudocode = {}
        for problem in tqdm(problems, desc="Generating pseudocode", unit="problem"):
            problem_name = problem["task_name"]

            # Check if source language is available for this problem
            if source_language not in problem["implementations"]:
                logger.warning(
                    (
                        f"Source language {source_language} not available for"
                        f" problem {problem_name}"
                    )
                )
                continue

            # Get source code
            source_code = problem["implementations"][source_language]

            # Create a safe directory and filename
            safe_task_name = sanitize_filename(problem_name)

            # Create problem-specific directory
            problem_dir = Path(pseudocode_output_dir) / safe_task_name
            problem_dir.mkdir(parents=True, exist_ok=True)

            # Generate pseudocode
            pseudocode = generate_pseudocode(
                problem_name=problem_name,
                problem_description=problem["task_description"],
                source_code=source_code,
                source_language=source_language,
                output_path=problem_dir / "pseudocode.txt",
                llm_api_url=self.llm_api_url,
            )

            all_pseudocode[problem_name] = pseudocode

            # Update experiment state
            if problem_name in self.experiment_state["problems"]:
                self.experiment_state["problems"][problem_name]["pseudocode"] = {
                    "source_language": source_language,
                    "content": pseudocode,
                }

        self.save_state()
        return all_pseudocode

    def translate_to_languages(
        self,
        problems: List[Dict[str, Any]],
        target_languages: List[str],
        pseudocode: Optional[Dict[str, str]] = None,
        source_language: Optional[str] = None,
        translations_output_dir: Optional[str] = None,
        use_pseudocode: bool = True,
        use_graphrag: bool = False,
    ) -> Dict[str, Dict[str, str]]:
        """
        Translate problems to target languages either from pseudocode or directly.

        Args:
            problems: List of problems to translate
            target_languages: List of target languages
            pseudocode: Optional dictionary of pseudocode by problem name
            source_language: Optional source language for direct translation
            translations_output_dir: Optional directory to save translations to
            use_pseudocode: Whether to use pseudocode as an intermediate step
            use_graphrag: Whether to use GraphRAG for enhanced context

        Returns:
            Dictionary mapping problem names to translations by language
        """
        from experiment.translator import translate_code

        # Initialize GraphRAG if requested
        graphrag_instance = None
        if use_graphrag:
            try:
                from utils.rag.rag import GraphRAG

                logger.info("Initializing GraphRAG for enhanced translation context")
                graphrag_instance = GraphRAG()
            except Exception as e:
                logger.error(f"Failed to initialize GraphRAG: {str(e)}")
                logger.warning("Continuing without GraphRAG enhancement")
                use_graphrag = False

        # Default translations directory is inside experiment output directory
        if translations_output_dir is None:
            translations_output_dir = self.output_dir / "translations"
        os.makedirs(translations_output_dir, exist_ok=True)

        if not use_pseudocode and source_language is None:
            raise ValueError(
                "Source language must be provided when not using pseudocode"
            )

        # Translate each problem to each target language
        all_translations = {}
        for problem in tqdm(problems, desc="Translating problems", unit="problem"):
            problem_name = problem["task_name"]

            # Create a safe directory name
            safe_task_name = sanitize_filename(problem_name)

            # Create the problem-specific directory
            problem_dir = Path(translations_output_dir) / safe_task_name
            problem_dir.mkdir(parents=True, exist_ok=True)

            problem_translations = {}

            # Create a progress bar for languages
            lang_progress = tqdm(
                target_languages,
                desc=f"Translating {problem_name}",
                unit="lang",
                leave=False,
            )

            for target_lang in lang_progress:
                lang_progress.set_description(f"Translating to {target_lang}")

                # Skip if target language is the same as source language
                if (
                    not use_pseudocode
                    and target_lang.lower() == source_language.lower()
                ):
                    continue

                if use_pseudocode:
                    # Use pseudocode as intermediate step
                    if pseudocode is None or problem_name not in pseudocode:
                        logger.warning(
                            f"Pseudocode not available for problem {problem_name}"
                        )
                        continue

                    source_code = pseudocode[problem_name]
                    source_lang = "pseudocode"
                else:
                    # Direct translation from source language
                    if source_language not in problem["implementations"]:
                        logger.warning(
                            (
                                f"Source language {source_language} not available"
                                f" for problem {problem_name}"
                            )
                        )
                        continue

                    source_code = problem["implementations"][source_language]
                    source_lang = source_language

                # Get extension from language config
                safe_lang_name = sanitize_filename(target_lang)
                ext = self.language_tools.get_language_config(target_lang.lower()).get(
                    "extension", target_lang.lower()
                )

                # Translate code
                translated_code = translate_code(
                    problem_name=problem_name,
                    problem_description=problem["task_description"],
                    source_code=source_code,
                    source_language=source_lang,
                    target_language=target_lang,
                    output_path=problem_dir / f"{safe_lang_name}.{ext}",
                    use_graphrag=use_graphrag,
                    graphrag_instance=graphrag_instance,
                    llm_api_url=self.llm_api_url,
                )

                problem_translations[target_lang] = translated_code

            all_translations[problem_name] = problem_translations

            # Update experiment state
            if problem_name in self.experiment_state["problems"]:
                self.experiment_state["translations"][problem_name] = (
                    problem_translations
                )

        self.save_state()
        return all_translations

    def evaluate_translations(
        self,
        problems: List[Dict[str, Any]],
        translations: Dict[str, Dict[str, str]],
        test_cases: Dict[str, List[Dict[str, Any]]],
        evaluation_output_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Evaluate translations against test cases.

        Args:
            problems: List of problems
            translations: Dictionary of translations by problem and language
            test_cases: Dictionary of test cases by problem
            evaluation_output_dir: Optional directory to save evaluation results to

        Returns:
            Dictionary mapping problem names to evaluation results by language
        """
        from experiment.evaluator import evaluate_translation

        # Default evaluation directory is inside experiment output directory
        if evaluation_output_dir is None:
            evaluation_output_dir = self.output_dir / "evaluation"
        os.makedirs(evaluation_output_dir, exist_ok=True)

        # Evaluate each translation
        all_results = {}
        for problem in tqdm(problems, desc="Evaluating translations", unit="problem"):
            problem_name = problem["task_name"]
            problem_dir = Path(evaluation_output_dir) / sanitize_filename(problem_name)
            problem_dir.mkdir(exist_ok=True)

            # Skip if problem has no translations or test cases
            if problem_name not in translations or problem_name not in test_cases:
                logger.warning(
                    (
                        f"Skipping evaluation for {problem_name} - missing"
                        f" translations or test cases"
                    )
                )
                continue

            problem_translations = translations[problem_name]
            problem_test_cases = test_cases[problem_name]

            problem_results = {}

            # Create a progress bar for languages
            lang_progress = tqdm(
                problem_translations.items(),
                desc=f"Evaluating {problem_name}",
                unit="lang",
                leave=False,
            )

            for lang, code in lang_progress:
                lang_progress.set_description(f"Evaluating {lang}")

                # Get language config
                lang_config = self.language_tools.get_language_config(lang.lower())
                if not lang_config:
                    logger.warning(f"No configuration found for language: {lang}")
                    continue

                # Get file extension
                extension = lang_config.get("extension", lang.lower())

                # Create source file
                source_file = problem_dir / f"{lang.lower()}.{extension}"
                with open(source_file, "w") as f:
                    f.write(code)

                # Evaluate translation
                eval_result = evaluate_translation(
                    source_file=source_file,
                    language=lang,
                    test_cases=problem_test_cases,
                    language_tools=self.language_tools,
                    output_path=problem_dir / f"{lang.lower()}_results.json",
                )

                problem_results[lang] = eval_result

                # Update experiment metrics
                if "compilation_success" not in self.experiment_state["metrics"]:
                    self.experiment_state["metrics"]["compilation_success"] = {}

                if "test_case_success" not in self.experiment_state["metrics"]:
                    self.experiment_state["metrics"]["test_case_success"] = {}

                if (
                    problem_name
                    not in self.experiment_state["metrics"]["compilation_success"]
                ):
                    self.experiment_state["metrics"]["compilation_success"][
                        problem_name
                    ] = {}

                if (
                    problem_name
                    not in self.experiment_state["metrics"]["test_case_success"]
                ):
                    self.experiment_state["metrics"]["test_case_success"][
                        problem_name
                    ] = {}

                self.experiment_state["metrics"]["compilation_success"][problem_name][
                    lang
                ] = eval_result["compiles"]

                if eval_result["test_results"]:
                    test_pass_rate = sum(
                        1 for tc in eval_result["test_results"] if tc["passed"]
                    ) / len(eval_result["test_results"])
                    self.experiment_state["metrics"]["test_case_success"][problem_name][
                        lang
                    ] = test_pass_rate

            all_results[problem_name] = problem_results

        self.save_state()
        return all_results

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a summary report of the experiment.

        Args:
            output_path: Optional path to save the report to

        Returns:
            Report as a string
        """
        from experiment.evaluator import generate_experiment_report

        # Default report path
        if output_path is None:
            output_path = self.output_dir / "experiment_report.md"

        # Generate report
        report = generate_experiment_report(
            experiment_state=self.experiment_state, output_path=output_path
        )

        return report

    def run_full_experiment(
        self,
        num_problems: int = 10,
        source_language: str = "C",
        target_languages: Optional[List[str]] = None,
        use_pseudocode: bool = True,
        require_all_languages: bool = False,
        use_cleaned_code: bool = True,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Run the full experiment pipeline.

        Args:
            num_problems: Number of problems to include
            source_language: Source language to translate from
            target_languages: List of target languages
                (defaults to all supported languages)
            use_pseudocode: Whether to use pseudocode as an intermediate step
            require_all_languages: If True, only include problems with implementations
                in all target languages
            use_cleaned_code: Whether to use LLM-cleaned code that removes examples
            batch_size: Number of samples to process in each batch for LLM cleaning

        Returns:
            Experiment metrics
        """
        logger.info(f"Starting full experiment with {num_problems} problems")

        # Use all supported languages if not specified
        if target_languages is None:
            target_languages = self.languages

        # 1. Generate test set
        logger.info("Generating test set...")
        problems = self.generate_test_set(
            num_problems=num_problems,
            require_all_languages=require_all_languages,
            use_cleaned_code=use_cleaned_code,
            batch_size=batch_size,
        )

        # 2. Generate test cases
        logger.info("Generating test cases...")
        test_cases = self.generate_test_cases(
            problems=problems, source_language=source_language
        )

        # Filter the problems list to only include those with successful test cases
        successful_problems = [p for p in problems if p["task_name"] in test_cases]
        if len(successful_problems) < len(problems):
            logger.info(
                (
                    f"Proceeding with {len(successful_problems)}/{len(problems)} "
                    f"problems after test case filtering"
                )
            )
            problems = successful_problems

        # 3. Generate pseudocode if needed
        pseudocode = None
        if use_pseudocode:
            logger.info(f"Generating pseudocode from {source_language}...")
            pseudocode = self.generate_pseudocode(
                problems=problems, source_language=source_language
            )

        # Store original experiment state to track both types of results
        self.experiment_state["metrics"]["regular"] = {
            "compilation_success": {},
            "test_case_success": {},
        }
        self.experiment_state["metrics"]["graphrag"] = {
            "compilation_success": {},
            "test_case_success": {},
        }
        self.experiment_state["translations"] = {"regular": {}, "graphrag": {}}

        # 4a. Translate to target languages using regular LLM
        logger.info(
            (f"Translating to {len(target_languages)} languages using regular LLM...")
        )
        # Create regular translation output directory
        regular_translations_dir = self.output_dir / "translations_regular"
        regular_translations = self.translate_to_languages(
            problems=problems,
            target_languages=target_languages,
            pseudocode=pseudocode,
            source_language=source_language,
            translations_output_dir=regular_translations_dir,
            use_pseudocode=use_pseudocode,
            use_graphrag=False,
        )

        # Store regular translations in experiment state
        self.experiment_state["translations"]["regular"] = regular_translations

        # 4b. Translate to target languages using GraphRAG
        logger.info(
            (
                f"Translating to {len(target_languages)} languages with "
                f"GraphRAG enhancement..."
            )
        )
        # Create GraphRAG translation output directory
        graphrag_translations_dir = self.output_dir / "translations_graphrag"
        graphrag_translations = self.translate_to_languages(
            problems=problems,
            target_languages=target_languages,
            pseudocode=pseudocode,
            source_language=source_language,
            translations_output_dir=graphrag_translations_dir,
            use_pseudocode=use_pseudocode,
            use_graphrag=True,
        )

        # Store GraphRAG translations in experiment state
        self.experiment_state["translations"]["graphrag"] = graphrag_translations

        # Create temporary backup of the metrics to avoid overwriting
        backup_metrics = self.experiment_state["metrics"].copy()

        # 5a. Evaluate regular translations
        logger.info("Evaluating regular translations...")
        self.experiment_state["metrics"]["compilation_success"] = backup_metrics[
            "regular"
        ]["compilation_success"]
        self.experiment_state["metrics"]["test_case_success"] = backup_metrics[
            "regular"
        ]["test_case_success"]

        regular_evaluation_dir = self.output_dir / "evaluation_regular"
        _ = self.evaluate_translations(
            problems=problems,
            translations=regular_translations,
            test_cases=test_cases,
            evaluation_output_dir=regular_evaluation_dir,
        )

        # Save regular metrics
        self.experiment_state["metrics"]["regular"]["compilation_success"] = (
            self.experiment_state["metrics"]["compilation_success"]
        )
        self.experiment_state["metrics"]["regular"]["test_case_success"] = (
            self.experiment_state["metrics"]["test_case_success"]
        )

        # 5b. Evaluate GraphRAG translations
        logger.info("Evaluating GraphRAG translations...")
        self.experiment_state["metrics"]["compilation_success"] = backup_metrics[
            "graphrag"
        ]["compilation_success"]
        self.experiment_state["metrics"]["test_case_success"] = backup_metrics[
            "graphrag"
        ]["test_case_success"]

        graphrag_evaluation_dir = self.output_dir / "evaluation_graphrag"
        _ = self.evaluate_translations(
            problems=problems,
            translations=graphrag_translations,
            test_cases=test_cases,
            evaluation_output_dir=graphrag_evaluation_dir,
        )

        # Save GraphRAG metrics
        self.experiment_state["metrics"]["graphrag"]["compilation_success"] = (
            self.experiment_state["metrics"]["compilation_success"]
        )
        self.experiment_state["metrics"]["graphrag"]["test_case_success"] = (
            self.experiment_state["metrics"]["test_case_success"]
        )

        # Restore the top-level metrics for the main report
        # (we'll combine the best of both for now)
        self.experiment_state["metrics"]["compilation_success"] = {
            problem: {
                lang: max(
                    self.experiment_state["metrics"]["regular"]["compilation_success"]
                    .get(problem, {})
                    .get(lang, False),
                    self.experiment_state["metrics"]["graphrag"]["compilation_success"]
                    .get(problem, {})
                    .get(lang, False),
                )
                for lang in set(
                    list(
                        self.experiment_state["metrics"]["regular"][
                            "compilation_success"
                        ]
                        .get(problem, {})
                        .keys()
                    )
                    + list(
                        self.experiment_state["metrics"]["graphrag"][
                            "compilation_success"
                        ]
                        .get(problem, {})
                        .keys()
                    )
                )
            }
            for problem in set(
                list(
                    self.experiment_state["metrics"]["regular"][
                        "compilation_success"
                    ].keys()
                )
                + list(
                    self.experiment_state["metrics"]["graphrag"][
                        "compilation_success"
                    ].keys()
                )
            )
        }

        self.experiment_state["metrics"]["test_case_success"] = {
            problem: {
                lang: max(
                    self.experiment_state["metrics"]["regular"]["test_case_success"]
                    .get(problem, {})
                    .get(lang, 0),
                    self.experiment_state["metrics"]["graphrag"]["test_case_success"]
                    .get(problem, {})
                    .get(lang, 0),
                )
                for lang in set(
                    list(
                        self.experiment_state["metrics"]["regular"]["test_case_success"]
                        .get(problem, {})
                        .keys()
                    )
                    + list(
                        self.experiment_state["metrics"]["graphrag"][
                            "test_case_success"
                        ]
                        .get(problem, {})
                        .keys()
                    )
                )
            }
            for problem in set(
                list(
                    self.experiment_state["metrics"]["regular"][
                        "test_case_success"
                    ].keys()
                )
                + list(
                    self.experiment_state["metrics"]["graphrag"][
                        "test_case_success"
                    ].keys()
                )
            )
        }

        self.save_state()

        # 6. Generate reports
        logger.info("Generating experiment reports...")
        # Overall report
        self.generate_report()

        # Generate separate reports for regular and GraphRAG
        from experiment.evaluator import generate_experiment_report

        # Create a copy of the experiment state for the regular report
        regular_state = self.experiment_state.copy()
        regular_state["metrics"]["compilation_success"] = regular_state["metrics"][
            "regular"
        ]["compilation_success"]
        regular_state["metrics"]["test_case_success"] = regular_state["metrics"][
            "regular"
        ]["test_case_success"]
        regular_report_path = self.output_dir / "regular_report.md"
        generate_experiment_report(regular_state, regular_report_path)

        # Create a copy of the experiment state for the GraphRAG report
        graphrag_state = self.experiment_state.copy()
        graphrag_state["metrics"]["compilation_success"] = graphrag_state["metrics"][
            "graphrag"
        ]["compilation_success"]
        graphrag_state["metrics"]["test_case_success"] = graphrag_state["metrics"][
            "graphrag"
        ]["test_case_success"]
        graphrag_report_path = self.output_dir / "graphrag_report.md"
        generate_experiment_report(graphrag_state, graphrag_report_path)

        # Generate comparison report
        self._generate_comparison_report()

        logger.info("Experiment completed successfully")
        return self.experiment_state["metrics"]

    def _generate_comparison_report(self) -> None:
        """
        Generate a comparison report between regular and GraphRAG results.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import polars as pl

        regular_metrics = self.experiment_state["metrics"]["regular"]
        graphrag_metrics = self.experiment_state["metrics"]["graphrag"]

        comparison_path = self.output_dir / "comparison_report.md"
        charts_dir = self.output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        # Calculate overall metrics for each approach
        regular_compilation = {}
        graphrag_compilation = {}
        regular_test_success = {}
        graphrag_test_success = {}

        # Compile language-specific metrics
        for _, langs in regular_metrics["compilation_success"].items():
            for lang, success in langs.items():
                if lang not in regular_compilation:
                    regular_compilation[lang] = []
                regular_compilation[lang].append(int(success))

        for _, langs in graphrag_metrics["compilation_success"].items():
            for lang, success in langs.items():
                if lang not in graphrag_compilation:
                    graphrag_compilation[lang] = []
                graphrag_compilation[lang].append(int(success))

        for _, langs in regular_metrics["test_case_success"].items():
            for lang, rate in langs.items():
                if lang not in regular_test_success:
                    regular_test_success[lang] = []
                regular_test_success[lang].append(float(rate))

        for _, langs in graphrag_metrics["test_case_success"].items():
            for lang, rate in langs.items():
                if lang not in graphrag_test_success:
                    graphrag_test_success[lang] = []
                graphrag_test_success[lang].append(float(rate))

        # Calculate averages
        regular_comp_avg = {
            lang: sum(rates) / len(rates) if rates else 0
            for lang, rates in regular_compilation.items()
        }

        graphrag_comp_avg = {
            lang: sum(rates) / len(rates) if rates else 0
            for lang, rates in graphrag_compilation.items()
        }

        regular_test_avg = {
            lang: sum(rates) / len(rates) if rates else 0
            for lang, rates in regular_test_success.items()
        }

        graphrag_test_avg = {
            lang: sum(rates) / len(rates) if rates else 0
            for lang, rates in graphrag_test_success.items()
        }

        # Generate report
        all_languages = sorted(
            set(
                list(regular_comp_avg.keys())
                + list(graphrag_comp_avg.keys())
                + list(regular_test_avg.keys())
                + list(graphrag_test_avg.keys())
            )
        )

        cols = [
            "Language",
            "Regular Compilation",
            "GraphRAG Compilation",
            "Difference",
            "Regular Test Success",
            "GraphRAG Test Success",
            "Difference",
        ]
        report = [
            "# GraphRAG vs Regular LLM Comparison Report",
            f"*Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "## Summary",
            "Regular LLM translations versus GraphRAG-enhanced translations.\n",
            "### Performance Comparison by Language\n",
            "|" + "|".join([f"| {col} " for col in cols]) + "|",
            "|" + "|".join([f"| {'-' * len(col)} " for col in cols]) + "|",
        ]

        for lang in all_languages:
            reg_comp = regular_comp_avg.get(lang, 0) * 100
            graph_comp = graphrag_comp_avg.get(lang, 0) * 100
            comp_diff = graph_comp - reg_comp

            reg_test = regular_test_avg.get(lang, 0) * 100
            graph_test = graphrag_test_avg.get(lang, 0) * 100
            test_diff = graph_test - reg_test

            comp_diff_str = f"{comp_diff:+.1f}%" if comp_diff != 0 else "0.0%"
            test_diff_str = f"{test_diff:+.1f}%" if test_diff != 0 else "0.0%"

            report.append(
                f"| {lang} | {reg_comp:.1f}% | {graph_comp:.1f}% | {comp_diff_str} | "
                f"{reg_test:.1f}% | {graph_test:.1f}% | {test_diff_str} |"
            )

        # Add overall averages
        all_reg_comp = [v for values in regular_compilation.values() for v in values]
        all_graph_comp = [v for values in graphrag_compilation.values() for v in values]
        all_reg_test = [v for values in regular_test_success.values() for v in values]
        all_graph_test = [
            v for values in graphrag_test_success.values() for v in values
        ]

        overall_reg_comp = (
            sum(all_reg_comp) / len(all_reg_comp) if all_reg_comp else 0
        ) * 100
        overall_graph_comp = (
            sum(all_graph_comp) / len(all_graph_comp) if all_graph_comp else 0
        ) * 100
        overall_reg_test = (
            sum(all_reg_test) / len(all_reg_test) if all_reg_test else 0
        ) * 100
        overall_graph_test = (
            sum(all_graph_test) / len(all_graph_test) if all_graph_test else 0
        ) * 100

        overall_comp_diff = overall_graph_comp - overall_reg_comp
        overall_test_diff = overall_graph_test - overall_reg_test

        comp_diff_str = (
            f"{overall_comp_diff:+.1f}%" if overall_comp_diff != 0 else "0.0%"
        )
        test_diff_str = (
            f"{overall_test_diff:+.1f}%" if overall_test_diff != 0 else "0.0%"
        )

        report.append("")
        report.append("### Overall Results\n")
        report.append("| Approach | Compilation Success | Test Case Success |")
        report.append("|----------|---------------------|-------------------|")
        report.append(
            f"| Regular LLM | {overall_reg_comp:.1f}% | {overall_reg_test:.1f}% |"
        )
        report.append(
            f"| GraphRAG LLM | {overall_graph_comp:.1f}% | {overall_graph_test:.1f}% |"
        )
        report.append(f"| Difference | {comp_diff_str} | {test_diff_str} |")

        # Generate comparison charts
        # Create DataFrame for visualization
        data = []
        for lang in all_languages:
            data.append(
                {
                    "Language": lang,
                    "Regular Compilation": regular_comp_avg.get(lang, 0) * 100,
                    "GraphRAG Compilation": graphrag_comp_avg.get(lang, 0) * 100,
                    "Regular Test Success": regular_test_avg.get(lang, 0) * 100,
                    "GraphRAG Test Success": graphrag_test_avg.get(lang, 0) * 100,
                    "Compilation Difference": (
                        graphrag_comp_avg.get(lang, 0) - regular_comp_avg.get(lang, 0)
                    )
                    * 100,
                    "Test Success Difference": (
                        graphrag_test_avg.get(lang, 0) - regular_test_avg.get(lang, 0)
                    )
                    * 100,
                }
            )

        df = pl.DataFrame(data)

        # Chart 1: Compilation success comparison
        plt.figure(figsize=(12, 6))
        width = 0.35
        x = np.arange(len(df))
        plt.bar(
            x - width / 2,
            df["Regular Compilation"],
            width,
            label="Regular LLM",
            color="blue",
        )
        plt.bar(
            x + width / 2,
            df["GraphRAG Compilation"],
            width,
            label="GraphRAG",
            color="green",
        )
        plt.title("Compilation Success Rate: Regular vs GraphRAG")
        plt.xlabel("Language")
        plt.ylabel("Success Rate (%)")
        plt.xticks(x, df["Language"], rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        comp_chart_path = charts_dir / "compilation_comparison.png"
        plt.savefig(comp_chart_path)
        plt.close()

        # Chart 2: Test case success comparison
        plt.figure(figsize=(12, 6))
        plt.bar(
            x - width / 2,
            df["Regular Test Success"],
            width,
            label="Regular LLM",
            color="blue",
        )
        plt.bar(
            x + width / 2,
            df["GraphRAG Test Success"],
            width,
            label="GraphRAG",
            color="green",
        )
        plt.title("Test Case Success Rate: Regular vs GraphRAG")
        plt.xlabel("Language")
        plt.ylabel("Success Rate (%)")
        plt.xticks(x, df["Language"], rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        test_chart_path = charts_dir / "test_success_comparison.png"
        plt.savefig(test_chart_path)
        plt.close()

        # Chart 3: Difference in performance (GraphRAG - Regular)
        plt.figure(figsize=(12, 6))
        plt.bar(
            x - width / 2,
            df["Compilation Difference"],
            width,
            label="Compilation Improvement",
            color="green" if overall_comp_diff > 0 else "red",
        )
        plt.bar(
            x + width / 2,
            df["Test Success Difference"],
            width,
            label="Test Success Improvement",
            color="green" if overall_test_diff > 0 else "red",
        )
        plt.title("Performance Improvement with GraphRAG (GraphRAG - Regular)")
        plt.xlabel("Language")
        plt.ylabel("Difference (%)")
        plt.axhline(y=0, color="k", linestyle="-")
        plt.xticks(x, df["Language"], rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        diff_chart_path = charts_dir / "performance_difference.png"
        plt.savefig(diff_chart_path)
        plt.close()

        # Chart 4: Overall comparison
        plt.figure(figsize=(10, 6))
        overall_data = [
            overall_reg_comp,
            overall_graph_comp,
            overall_reg_test,
            overall_graph_test,
        ]
        categories = [
            "Regular\nCompilation",
            "GraphRAG\nCompilation",
            "Regular\nTest Success",
            "GraphRAG\nTest Success",
        ]
        colors = ["blue", "green", "blue", "green"]
        plt.bar(categories, overall_data, color=colors)
        plt.title("Overall Performance Comparison")
        plt.ylabel("Success Rate (%)")
        plt.tight_layout()
        overall_chart_path = charts_dir / "overall_comparison.png"
        plt.savefig(overall_chart_path)
        plt.close()

        # Add charts to the report
        report.append("\n## Comparison Charts\n")
        report.append("### Compilation Success Comparison\n")
        report.append(
            "![Compilation Success Comparison](charts/compilation_comparison.png)\n"
        )

        report.append("### Test Case Success Comparison\n")
        report.append(
            "![Test Case Success Comparison](charts/test_success_comparison.png)\n"
        )

        report.append("### Performance Difference (GraphRAG - Regular)\n")
        report.append("![Performance Difference](charts/performance_difference.png)\n")

        report.append("### Overall Performance Comparison\n")
        report.append("![Overall Performance](charts/overall_comparison.png)\n")

        # Write the report
        with open(comparison_path, "w") as f:
            f.write("\n".join(report))

        logger.info(f"Saved comparison report with charts to {comparison_path}")
