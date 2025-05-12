"""
Code translation module for code translation experiments.
"""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger("experiment.translator")

# Default LLM API endpoint (local server)
DEFAULT_LLM_API = "http://localhost:1234/v1/chat/completions"


def generate_pseudocode(
    problem_name: str,
    problem_description: str,
    source_code: str,
    source_language: str,
    output_path: Optional[str] = None,
    llm_api_url: str = DEFAULT_LLM_API,
) -> str:
    """
    Generate pseudocode for a problem from source language.

    Args:
        problem_name: Name of the problem
        problem_description: Description of the problem
        source_code: Source code in the original language
        source_language: Source language name
        output_path: Optional path to save pseudocode to
        llm_api_url: URL for the LLM API server

    Returns:
        Generated pseudocode
    """
    logger.info(f"Generating pseudocode for {problem_name} from {source_language}")

    # Prepare prompt for the LLM
    system_prompt = """
    You are a helpful programming assistant that excels at creating
    clear, language-agnostic pseudocode. Your task is to analyze
    source code in a specific programming language and create detailed pseudocode
    that captures the core algorithm and logic, but without language-specific
    syntax or features.\n
    The pseudocode should:
    1. Be easy to understand for programmers of any background
    2. Preserve the algorithmic approach and logic of the original code
    3. Explain key data structures and their purpose
    4. Include clear control flow and conditionals
    5. Document any edge cases or special handling
    6. Be detailed enough that it could be implemented in any programming language\n
    For functions that operate on fixed values
    (like formatting functions that don't take input):
    1. Clearly indicate in the pseudocode that the function uses hard-coded values
    2. Explain what the expected output should be
    3. Detail how the formatting or processing is done to produce that output\n
    Use standard pseudocode conventions with clear indentation and structure.
    """

    user_prompt = f"""
    Problem Name: {problem_name}\n
    Problem Description:
    {problem_description}\n
    Source Code ({source_language}):
    ```{source_language}
    {source_code}
    ```\n
    Please create detailed, language-agnostic pseudocode that captures
    the algorithm and logic of this code. Make the pseudocode clear
    and comprehensive enough that it could be translated to any programming language.\n
    IMPORTANT: If this code operates on fixed values without requiring input
    (like a formatting function), clearly indicate this in the pseudocode and
    specify the expected output.
    """

    try:
        # Prepare the request to the local LLM server
        payload = {
            "model": "any",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 3000,
            "temperature": 0.1,
        }

        # Send request to local LLM server
        response = requests.post(llm_api_url, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors

        response_data = response.json()
        pseudocode = response_data["choices"][0]["message"]["content"]

        logger.info(
            f"Generated pseudocode for {problem_name} ({len(pseudocode.split())} words)"
        )

        # Save pseudocode if output path is specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(pseudocode)
            logger.info(f"Saved pseudocode to {output_path}")

        return pseudocode

    except Exception as e:
        logger.error(f"Error generating pseudocode: {str(e)}")
        return f"Error generating pseudocode: {str(e)}"


def translate_code(
    problem_name: str,
    problem_description: str,
    source_code: str,
    source_language: str,
    target_language: str,
    output_path: Optional[str] = None,
    use_graphrag: bool = False,
    graphrag_instance=None,
    llm_api_url: str = DEFAULT_LLM_API,
) -> str:
    """
    Translate code from one language to another.

    Args:
        problem_name: Name of the problem
        problem_description: Description of the problem
        source_code: Source code to translate
        source_language: Source language name
        target_language: Target language name
        output_path: Optional path to save translated code to
        use_graphrag: Whether to use GraphRAG for enhanced context
        graphrag_instance: Optional pre-initialized GraphRAG instance
        llm_api_url: URL for the LLM API server

    Returns:
        Translated code
    """
    logger.info(
        f"Translating {problem_name} from {source_language} to {target_language}"
    )

    # Prepare prompt for the LLM
    system_prompt = """
    You are an expert polyglot programmer specializing in translating code
    between programming languages. Your task is to translate source code
    from one language to another while preserving the algorithm,
    logic, and functionality.\n
    When translating code, you should:
    1. Follow idiomatic practices of the target language
    2. Preserve the algorithmic approach and logic
    3. Use appropriate data structures in the target language
    4. Ensure error handling is maintained
    5. Add clear comments where necessary to explain complex logic
    6. Make the code fully functional and ready to run\n
    Pay special attention to functions that don't take any input but simply produce
    a fixed output (like formatting functions).
    In these cases, make sure your translation
    produces the exact same output as the original code when run without arguments.\n
    Provide ONLY the translated code, without explanations or
    descriptions outside the code itself. If explanation is needed,
    include it as comments within the code.
    """

    # Get GraphRAG context if requested
    graphrag_context = ""
    if use_graphrag and graphrag_instance:
        try:
            # Formulate a context query based on the problem and languages
            context_query = (
                f"How to implement {problem_name} in {target_language}? "
                f"Show code examples and best practices."
            )

            # Add source language for reference
            if source_language.lower() != "pseudocode":
                context_query += f" Compare with {source_language} implementation."

            # Get context from GraphRAG
            logger.info(
                f"Retrieving GraphRAG context for {target_language} implementation"
            )
            context_results = graphrag_instance.query(
                context_query,
                top_k=3,
                filters={"subject": target_language.lower()},
                rerank=True,
            )

            # Format the context for the prompt
            if "passages" in context_results and context_results["passages"]:
                graphrag_context = (
                    "Relevant documentation and examples for reference:\n\n"
                )
                for i, passage in enumerate(context_results["passages"]):
                    title = passage.get("title", "Untitled")
                    text = passage.get("text", "")
                    graphrag_context += f"[{i + 1}] {title}\n{text}\n\n"

                logger.info(
                    (
                        f"Retrieved {len(context_results['passages'])}"
                        f" relevant passages from documentation"
                    )
                )
            else:
                logger.warning(
                    f"No relevant GraphRAG context found for {target_language}"
                )
        except Exception as e:
            logger.error(f"Error retrieving GraphRAG context: {str(e)}")
            # Continue without GraphRAG context if there's an error

    try:
        # Build the user prompt
        user_prompt = f"""
        Task: {problem_name}
        Problem description: {problem_description}\n
        Source language: {source_language}\n
        Source code:
        ```{source_language}
        {source_code}
        ```
        Target language: {target_language}\n
        {graphrag_context}
        Translate the above code to {target_language}.\n
        IMPORTANT: Analyze whether this is a function that operates on fixed values
        without requiring any user input. If so, make sure your translation produces
        the exact same formatted output as the original code would.\n

        Please provide only the resulting code in {target_language}.
        """

        # Prepare the request to the local LLM server
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4000,
            "temperature": 0.2,
        }

        # Send request to local LLM server
        response = requests.post(llm_api_url, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors

        response_data = response.json()
        translated_code = response_data["choices"][0]["message"]["content"]

        # Clean up the response if it contains markdown code blocks
        if (
            f"```{target_language.lower()}" in translated_code.lower()
            or "```" in translated_code
        ):
            # Extract just the code block
            code_blocks = []
            in_code_block = False
            code_block_lines = []

            for line in translated_code.split("\n"):
                if line.strip().startswith("```"):
                    if in_code_block:
                        code_blocks.append("\n".join(code_block_lines))
                        code_block_lines = []
                    in_code_block = not in_code_block
                    continue

                if in_code_block:
                    code_block_lines.append(line)

            if code_block_lines:  # In case the last code block wasn't closed
                code_blocks.append("\n".join(code_block_lines))

            if code_blocks:
                # Use the longest code block (most likely the complete translation)
                translated_code = max(code_blocks, key=len)
            else:
                # Fall back to the full response if no code blocks found
                translated_code = translated_code.replace("```", "").strip()

        logger.info(
            (
                f"Translated {problem_name} to {target_language}",
                f" ({len(translated_code.split())} words)",
            )
        )

        # Save translated code if output path is specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(translated_code)
            logger.info(f"Saved translated code to {output_path}")

        return translated_code

    except Exception as e:
        logger.error(f"Error translating code: {str(e)}")
        return f"Error translating code: {str(e)}"
