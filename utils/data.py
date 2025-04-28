import polars as pl
from pathlib import Path

DATASET = 'hf://datasets/christopher/rosetta-code/data/train-00000-of-00001-8b4da49264116bbf.parquet'
CACHE_DIR = Path('cache')
BLACKLISTED_LANGUAGES = [
    '3. Output configuration', 
    'OS X sha256sum', 
    'Alternative version', 
    'உயிர்/Uyir', 
    '2. Calendar data functions',
    'Solution with recursion',
    '1. Grid structure functions',
    'Alternate version to handle 64 and 128 bit integers.',
    'Programming Language',
    'Writing your first program',
    'Installing Processing'
]
RELABELED_LANGUAGES = {
    'Plain English': [
        'Detailed Description of Programming Task'
    ],
    'Mathematica': [
        'Wolfram Language', 
        'Wolframalpha', 
        'Mathematica / Wolfram Language', 
        'Mathematica/Wolfram Language',
        'Mathematica//Wolfram Language',
        'Mathematica/ Wolfram Language',
        'Wolfram Language/Mathematica'
    ],
    'Python 3': [
        'Python 3.x Long Form'
    ],
    'Extended BrainFuck': [
        'Extended Brainf***'
    ],
    'Brainfuck': [
        'Brainf***'
    ],
    'Fish': [
        'friendly interactive shell'
    ],
    'VBA': [
        'VBA (Visual Basic for Application)'
    ],
    'Batch': [
        'Batch File'
    ],
    'Vlang': [
        '{{header|Vlang}'
    ],
    'VBScript': [
        'vbscript',
        'vbScript'
    ],
    'ooREXX': [
        'ooRexx'
    ],
    'UNIX Shell': [
        'Unix shell',
        'Unix Shell',
        'Shell'
    ],
    'Bash': [
        'Bash (Feat. Sed & Tr)',
        'Bash Shell'
    ],
    'TypeScript': [
        'Typescript'
    ],
    'Transact SQL': [
        'Transact-SQL',
        'Transact-SQL (MSSQL)',
        'TSQL'
    ],
    'Tcl': [
        'Tcl+SQLite'
    ],
    'TI-83 BASIC, TI-89 BASIC': [
        'TI-83 BASIC',
        'TI-83_BASIC',
        'TI-89 BASIC'
    ]
}

def get_dataset():
    print(RELABELED_LANGUAGES['Vlang'])
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(CACHE_DIR / 'dataset.parquet').exists():
        data = pl.read_parquet(DATASET)
        full_languages = data['language_name'].unique().len()
        data = data.filter(~pl.col('language_name').is_in(BLACKLISTED_LANGUAGES))
        post_blacklist = data['language_name'].unique().len()
        print(f"Removed {full_languages - post_blacklist} Blacklisted rows (Requested {len(BLACKLISTED_LANGUAGES)})")
        data = data.with_columns(pl.col('language_name').str.to_titlecase())
        post_titlecase = data['language_name'].unique().len()
        print(f"Converted to titlecase. Reduced number of languages by {post_blacklist - post_titlecase}")
        for new, old in RELABELED_LANGUAGES.items():
            data = data.with_columns(
                pl.when(pl.col('language_name').is_in(old))
                .then(pl.lit(new))
                .otherwise(pl.col('language_name'))
                .alias('language_name')
            )
        post_relabel = data['language_name'].unique().len()
        print(f"Reduced number of languages by {post_titlecase - post_relabel} through relabeling (Requested {len([x for y in RELABELED_LANGUAGES.values() for x in y]) - len(RELABELED_LANGUAGES)})")
        data.write_parquet(CACHE_DIR / 'dataset.parquet')
    else:
        data = pl.read_parquet(CACHE_DIR / 'dataset.parquet')

    return data