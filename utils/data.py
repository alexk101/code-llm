import re
from pathlib import Path
from typing import Dict, List

import polars as pl
import requests
from tqdm import tqdm

from utils.definitions import (
    BLACKLISTED_LANGUAGES,
    CACHE_DIR,
    CODE_COLS,
    DATASET,
    ID_COLS,
    RELABELED_LANGUAGES,
    TOP_LANGUAGES,
)
from utils.misc import cache_data


def is_python2_code(code: str) -> bool:
    """
    Detects if the given code is Python 2 by looking for common
    Python 2 patterns.

    Args:
        code: The Python code to analyze

    Returns:
        True if the code appears to be Python 2, False otherwise
    """
    # Skip empty code
    if not code or len(code.strip()) == 0:
        return False

    # Common Python 2 indicators
    indicators = [
        # Print statement without parentheses (most reliable indicator)
        r"^\s*print\s+[^(]",
        # xrange instead of range
        r"\bxrange\s*\(",
        # raw_input() instead of input()
        r"\braw_input\s*\(",
        # except Exception, e: syntax
        r"except\s+\w+\s*,\s*\w+:",
        # <> operator (instead of !=)
        r"<>",
        # unicode literals
        r"\bunicode\s*\(",
        # iteritems, iterkeys, itervalues methods
        r"\.iter(items|keys|values)\s*\(",
        # raw_input
        r"\braw_input\s*\(",
    ]

    for pattern in indicators:
        if re.search(pattern, code, re.MULTILINE):
            return True

    # Check for Python 2 style imports
    py2_imports = [
        "import urllib2",
        "import Queue",
        "import ConfigParser",
        "import HTMLParser",
        "import SocketServer",
        "import SimpleHTTPServer",
        "import httplib",
    ]

    for imp in py2_imports:
        if imp in code:
            return True

    return False


def clean_code_with_llm(
    code: str,
    language: str,
    task_name: str,
    llm_api_url: str = "http://localhost:1234/v1/chat/completions",
) -> str:
    """
    Use an LLM to clean and format code, removing unnecessary examples while
    preserving the algorithm's functionality.

    Args:
        code: The source code to clean
        language: The programming language of the code
        task_name: The name of the task/problem the code implements
        llm_api_url: URL for the LLM API endpoint

    Returns:
        Cleaned code that implements only the core algorithm
    """
    # Skip if code is empty or too short
    if not code or len(code.strip()) < 30:
        return code

    # Create prompt for the LLM
    prompt = f"""
You are a software engineer tasked with cleaning and standardizing code from the
Rosetta Code project. The following code is a {language} implementation of
the "{task_name}" task.

Your job is to clean this code by:
1. Removing unnecessary examples, comments, and testing code
2. Keeping only the core algorithm implementation
3. Formatting the code according to {language} best practices
4. Ensuring the code is compilable and functional
5. Preserving the exact algorithm and approach
(DO NOT change the algorithm or functionality)

Provide ONLY the cleaned code as your response, with no additional explanations.

If the code is meant to operate on some kind of input, ensure that the code is written
in a way that allows it to be run with input from the command line. The input will just
be the values for the variables, separated by spaces.

Example inputs:
python my_script.py 1 2 3
./binary_search 1 2 3

Original code:
```{language}
{code}
```
"""

    try:
        # Prepare the request for the LLM API
        payload = {
            "model": "any",  # The server will use whatever model it has
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a skilled code formatter and cleaner that "
                        "preserves algorithm functionality."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,  # Low temperature for more consistent results
            "max_tokens": 2000,
        }

        # Make the API call
        response = requests.post(
            llm_api_url, json=payload, headers={"Content-Type": "application/json"}
        )

        # Check if request was successful
        response.raise_for_status()

        # Extract the LLM's response
        result = response.json()
        cleaned_code = result["choices"][0]["message"]["content"].strip()

        # Extract code from markdown code blocks if present
        if cleaned_code.startswith("```") and cleaned_code.endswith("```"):
            # Extract code from markdown code block
            code_block_pattern = r"```(?:\w+)?\n([\s\S]*?)\n```"
            match = re.search(code_block_pattern, cleaned_code)
            if match:
                cleaned_code = match.group(1)

        return cleaned_code

    except Exception as e:
        print(f"Error cleaning code with LLM: {str(e)}")
        # Return original code if cleaning fails
        return code


def batch_clean_code(
    codes: List[Dict],
    llm_api_url: str = "http://localhost:1234/v1/chat/completions",
    batch_size: int = 100,
) -> List[Dict]:
    """
    Clean a batch of code samples using the LLM.

    Args:
        codes: List of dictionaries containing code samples with metadata
        llm_api_url: URL for the LLM API endpoint
        batch_size: Number of samples to process before updating progress

    Returns:
        List of dictionaries with cleaned code
    """
    cleaned_codes = []
    total = len(codes)

    print(f"Cleaning {total} code samples...")

    # Process in smaller batches to avoid overwhelming the LLM API
    for i in range(0, total, batch_size):
        batch = codes[i : min(i + batch_size, total)]

        desc = f"{i // batch_size + 1}/{((total + batch_size - 1) // batch_size)}"
        # Use tqdm for a nice progress bar for this batch
        for item in tqdm(
            batch,
            desc=f"Cleaning batch {desc}",
            unit="sample",
        ):
            # Clean the code
            cleaned_code = clean_code_with_llm(
                code=item["code"],
                language=item["language_name"],
                task_name=item["task_name"],
                llm_api_url=llm_api_url,
            )

            # Update the item with cleaned code
            item_copy = item.copy()
            item_copy["code"] = cleaned_code
            cleaned_codes.append(item_copy)

    print(f"Finished cleaning {total} code samples")
    return cleaned_codes


@cache_data(cache_path="dataset.parquet")
def get_dataset():
    data = pl.read_parquet(DATASET)
    full_languages = data["language_name"].unique().len()

    # Remove blacklisted languages
    data = data.filter(~pl.col("language_name").is_in(BLACKLISTED_LANGUAGES))
    post_blacklist = data["language_name"].unique().len()
    output = (
        f"Removed {full_languages - post_blacklist} Blacklisted rows "
        f"(Requested {len(BLACKLISTED_LANGUAGES)})"
    )
    print(output)

    # Relabel languages
    for new, old in RELABELED_LANGUAGES.items():
        data = data.with_columns(
            pl.when(pl.col("language_name").is_in(old))
            .then(pl.lit(new))
            .otherwise(pl.col("language_name"))
            .alias("language_name")
        )
    post_relabel = data["language_name"].unique().len()
    requested = len([x for y in RELABELED_LANGUAGES.values() for x in y]) - len(
        RELABELED_LANGUAGES
    )
    output = (
        f"Reduced number of languages by {post_blacklist - post_relabel} "
        f"through relabeling (Requested {requested})"
    )
    print(output)

    # Capitalize
    data = data.with_columns(pl.col("language_name").str.to_titlecase())
    post_titlecase = data["language_name"].unique().len()
    output = (
        f"Converted to titlecase. Reduced number of languages by "
        f"{post_relabel - post_titlecase}"
    )
    print(output)

    # Filter out Python 2 implementations
    pre_py2_filter = data.filter(pl.col("language_name") == "Python").height

    # Apply the filter
    data = data.filter(
        pl.when(pl.col("language_name") == "Python")
        .then(~pl.col("code").map_elements(is_python2_code))
        .otherwise(pl.lit(True))
    )

    post_py2_filter = data.filter(pl.col("language_name") == "Python").height
    output = f"Removed {pre_py2_filter - post_py2_filter} Python 2 implementations"
    print(output)

    return data


@cache_data(cache_path="language_info.parquet")
def get_language_info():
    """
    Get the language info from the top languages programming languages by TIOBE index.
    """
    df = pl.read_csv(TOP_LANGUAGES).with_columns(
        pl.col("Programming Language").str.to_titlecase()
    )
    if "Fotran" in df["Programming Language"].to_list():
        df = df.with_columns(
            pl.col("Programming Language").str.replace("Fotran", "Fortran")
        )
    if "Assembly Language" in df["Programming Language"].to_list():
        df = df.with_columns(
            pl.col("Programming Language").str.replace(
                "Assembly Language", "Assembly (X86)"
            )
        )
    if "Delphi/Object Pascal" in df["Programming Language"].to_list():
        df = df.with_columns(
            pl.col("Programming Language").str.replace("Delphi/Object Pascal", "Pascal")
        )

    # remove scratch
    remove_languages = ["Scratch", "Sql", "Visual Basic"]
    df = df.filter(~pl.col("Programming Language").is_in(remove_languages))
    return df


def narrow_problems(df, desired_languages=15, limit=0):
    """
    Narrow the dataset to only include problems that have at least
    a certain number of languages.
    """
    original_height = df.height
    print(f"Narrowing problems to at least {desired_languages} languages")
    languages = df["language_name"].unique().to_list()
    tasks_to_keep = []
    for task in df["task_name"].unique().to_list():
        task_df = df.filter(pl.col("task_name") == task)
        missing_languages = set(languages) - set(
            task_df["language_name"].unique().to_list()
        )
        if len(languages) - len(missing_languages) > desired_languages:
            tasks_to_keep.append(task)
        if limit and len(tasks_to_keep) >= limit:
            break

    filtered_df = df.filter(pl.col("task_name").is_in(tasks_to_keep))
    print(f"Tasks with more than {desired_languages} languages: {len(tasks_to_keep)}")
    print(f"Narrowed dataset to {filtered_df.height} rows from {original_height} rows")
    print(f"Removed {original_height - filtered_df.height} rows")
    return filtered_df


def generate_test_set(
    languages=None,
    use_cleaned_code=True,
    llm_api_url: str = "http://localhost:1234/v1/chat/completions",
    batch_size: int = 100,
    narrow=True,
    desired_languages=14,
    limit=0,
):
    """
    Generate a filtered dataset from Rosetta Code with implementations in
    selected languages.

    This function:
    1. Gets information about top programming languages from TIOBE index
    2. Gets the Rosetta Code dataset (with Python 2 implementations removed)
    3. Filters the dataset to include only implementations in the specified languages
    4. Removes duplicate implementations
    5. Optionally uses LLM to clean code AFTER filtering (more efficient)

    Args:
        languages: Optional list of specific languages to include. If None,
            uses top languages from TIOBE.
        use_cleaned_code: Whether to use the LLM-cleaned code (default: True)
        llm_api_url: URL for the LLM API endpoint
        batch_size: Number of samples to process in each batch for LLM cleaning

    Returns:
        A polars DataFrame with the filtered dataset ready for experiments
    """
    tiobe = get_language_info()
    rosettacode = get_dataset()

    # Select languages to include
    if languages is None:
        # Use top languages from TIOBE if no specific languages provided
        selected_languages = tiobe["Programming Language"].to_list()
        print(f"Using {len(selected_languages)} top languages from TIOBE index")
    else:
        # Use the provided languages list
        selected_languages = languages
        print(f"Using {len(selected_languages)} specified languages")

    # Get implementations in selected languages from Rosetta Code
    rosetta_languages = rosettacode.filter(
        pl.col("language_name").is_in(selected_languages)
    )
    rosetta_languages = rosetta_languages.drop("task_url", "language_url")

    # Remove duplicate implementations
    pre_clean = rosetta_languages.height
    rosetta_languages = rosetta_languages.group_by(ID_COLS).agg(
        pl.col(CODE_COLS).first()
    )
    post_clean = rosetta_languages.height
    print(f"Removed {pre_clean - post_clean} rows (Duplicate Implementations)")

    if narrow:
        rosetta_languages = narrow_problems(rosetta_languages, desired_languages, limit)

    # Now that we have filtered the dataset, optionally clean the code with LLM
    if use_cleaned_code:
        print(
            f"Cleaning code for {rosetta_languages.height} filtered implementations..."
        )

        # Check if we have a cached cleaned version of this exact filtered dataset
        cache_key = Path("cleaned_filtered_dataset.parquet")
        cache_key = CACHE_DIR / cache_key
        if cache_key.exists():
            # Try to load from cache
            cleaned_data = pl.read_parquet(cache_key)
            print(f"Loaded clean dataset from cache: {cleaned_data.height} entries")
        else:
            # If not cached, clean the filtered dataset
            data_dicts = rosetta_languages.to_dicts()

            # Clean the code in batches
            cleaned_data_dicts = batch_clean_code(
                data_dicts, llm_api_url=llm_api_url, batch_size=batch_size
            )

            # Convert back to polars DataFrame
            cleaned_data = pl.from_dicts(cleaned_data_dicts)

            # Cache this cleaned filtered dataset for future use
            try:
                cleaned_data.write_parquet(cache_key)
                print(f"Cached cleaned filtered dataset: {cleaned_data.height} entries")
            except Exception as e:
                print(f"Warning: Could not cache cleaned dataset: {e}")
        return cleaned_data

    # Return the filtered dataset without cleaning
    return rosetta_languages
