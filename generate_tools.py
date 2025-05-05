import polars as pl
import requests

from utils.data import get_dataset

ID_COLS = ["language_name", "task_name"]
CODE_COLS = ["code", "task_description"]


def get_language_info():
    url = "https://raw.githubusercontent.com/toUpperCase78/tiobe-index-ratings/refs/heads/master/Tiobe_Index_April2025.csv"
    response = requests.get(url)
    df = pl.read_csv(response.content).with_columns(
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

    # remove scratch
    df = df.filter(pl.col("Programming Language") != "Scratch")
    return df


def generate_test_set():
    df = get_dataset()

    # Get all languages
    languages = df["language_name"].value_counts().sort("count", descending=True)

    tiobe = get_language_info()

    rosettacode = pl.read_parquet("cache/dataset.parquet")
    print(rosettacode)

    # Select top 20 languages
    languages = tiobe["Programming Language"].to_list()

    # Get all languages from rosetta code
    rosetta_languages = rosettacode.filter(pl.col("language_name").is_in(languages))
    rosetta_languages = rosetta_languages.drop("task_url", "language_url")

    # Remove duplicate implementations
    pre_clean = rosetta_languages.height
    rosetta_languages = rosetta_languages.group_by(ID_COLS).agg(
        pl.col(CODE_COLS).first()
    )
    post_clean = rosetta_languages.height
    print(f"Removed {pre_clean - post_clean} rows (Duplicate Implementations)")

    return rosetta_languages


if __name__ == "__main__":
    rosetta_languages = generate_test_set()
    print(rosetta_languages)
