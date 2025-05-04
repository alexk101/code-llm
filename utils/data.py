from pathlib import Path

import polars as pl
import yaml

DATASET = "hf://datasets/christopher/rosetta-code/data/train-00000-of-00001-8b4da49264116bbf.parquet"
CACHE_DIR = Path("cache")
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
]


def load_relabeled_languages():
    with open("relabel_langs.yaml", "r") as file:
        return yaml.safe_load(file)


RELABELED_LANGUAGES = load_relabeled_languages()


def get_dataset(cache: bool = True):
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(CACHE_DIR / "dataset.parquet").exists():
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
        if cache:
            data.write_parquet(CACHE_DIR / "dataset.parquet")
    else:
        data = pl.read_parquet(CACHE_DIR / "dataset.parquet")

    return data


def main():
    data = get_dataset()
    print(data)


if __name__ == "__main__":
    main()
