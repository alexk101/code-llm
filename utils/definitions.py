from pathlib import Path

import yaml

# utils/data.py
DATASET = "hf://datasets/christopher/rosetta-code/data/train-00000-of-00001-8b4da49264116bbf.parquet"
TOP_LANGUAGES = "https://raw.githubusercontent.com/toUpperCase78/tiobe-index-ratings/refs/heads/master/Tiobe_Index_April2025.csv"
RELABELED_LANGUAGES = yaml.safe_load(open("relabel_langs.yaml", "r"))
BLACKLISTED_LANGUAGES = [
    "3. Output configuration",
    "OS X sha256sum",
    "Alternative version",
    "உயிர்/Uyir",
    "2. Calendar data functions",
    "Solution with recursion",
    "1. Grid structure functions",
    "Alternate version to handle 64 and 128 bit integers.",
    "Programming Language",
    "Writing your first program",
    "Installing Processing",
    "Run Basic",
    "Rust",
]
BLACKLISTED_PROBLEMS = [
    "hello, world!",
    "sleep",
    "input loop",
    "create a file",
    "read a file line by line",
    "user input/text",
    "hello world/graphical",
    "hello world/standard error",
    "hello world/text",
    "empty program",
    "empty string",
    "delete a file",
    "loops/infinite",
    "csv data manipulation",
    "menu",
    "show the epoch",
]

# utils/misc.py
CACHE_DIR = Path("cache")


# utils/validate_tools.py
ID_COLS = ["language_name", "task_name"]
CODE_COLS = ["code", "task_description"]

TIMEOUT = 10
