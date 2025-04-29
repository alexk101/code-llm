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
    'Installing Processing',
    'Run Basic'
]
RELABELED_LANGUAGES = {
    'Assembly (x86)': [
        'x86-64 Assembly',
        'x86 Assembly',
        'X86 Assembly',
        'X86-64 Assembly',
        'Assembly'
    ],
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
    'Python': [
        'Python 3.x Long Form',
        'Python 3',
        'Python+SQlite'
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
    'Visual Basic': [
        'Vba (Visual Basic For Application)',
        'VBA (Visual Basic for Application)',
        'VBA',
        'vba/Visual Basic',
        'Vba/Visual Basic',
        'VBA/Visual Basic',
        'vba',
        'Vba'
    ],
    'Batch': [
        'Batch File',
        'Batch file'
    ],
    'Vlang': [
        r'{{header|Vlang}'
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
        'Transact Sql',
        'Transact-SQL',
        'transact-Sql',
        'Transact-SQL (MSSQL)',
        'Transact-Sql (MSSQL)',
        'TSQL',
        'tsql'
    ],
    'Tcl': [
        'tcl+SQLite',
        'Tcl+SQLite'
    ],
    'TI-83 BASIC, TI-89 BASIC': [
        'TI-83 BASIC',
        'TI-83_BASIC',
        'TI-89 BASIC',
        'ti-89 BASIC',
        'ti-83 BASIC'
        'ti-83_BASIC'
        'ti-83 BASIC, ti-89 BASIC'
    ],
    'Zoea': [
        'Zoea Visual'
    ],
    'Oxygenbasic': [
        'Oxygenbasic X86 Assembler'
    ],
    'C/C++': [
        'C and C++',
        'C / C++'
    ],
    'C++': [
        'C++/CLI'
    ],
    'C': [
        'C Shell'
    ],
    'C#': [
        'c_sharp',
        'C# and Visual Basic .NET'
    ],
    'PHP': [
        'php',
        'PHP+SQLite'
    ],
    'Swift': [
        'Swift Playground'
    ],
    'SQL': [
        'SQL PL',
        'SQL/PostgreSQL',
        'PostgreSQL',
        'PL/pgSQL',
        'PL/SQL'
    ],
    'QuickBASIC': [
        'Quick BASIC',
        'Quick Basic/QBASIC/PDS 7.1/VB-DOS',
        'QuickBASIC 4.5'
    ],
    'PureBasic': [
        'PureBasic+SQLite'
    ],
    'PowerShell': [
        'PowerShell+SQLite'
    ],
    'Delphi/Object Pascal': [
        'Pascal / Delphi / Free Pascal',
        'Pascal and Object Pascal',
        'Delphi/Pascal',
        'Delphi and Pascal',
        'Pascal'
    ],
    'PARI/GP': [
        'PARIGP'
    ],
    'Matlab': [
        'Matlab/Octave',
        'MATLAB / Octave'
    ],
    'Javascript': [
        'JavaScript + HTML',
        'JavaScript + SVG',
        'Javascript/NodeJS'
    ]
}

def get_dataset(cache: bool = True):
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(CACHE_DIR / 'dataset.parquet').exists():
        data = pl.read_parquet(DATASET)
        full_languages = data['language_name'].unique().len()
        # Remove blacklisted languages
        data = data.filter(~pl.col('language_name').is_in(BLACKLISTED_LANGUAGES))
        post_blacklist = data['language_name'].unique().len()
        print(f"Removed {full_languages - post_blacklist} Blacklisted rows (Requested {len(BLACKLISTED_LANGUAGES)})")
        # Relabel languages
        for new, old in RELABELED_LANGUAGES.items():
            data = data.with_columns(
                pl.when(pl.col('language_name').is_in(old))
                .then(pl.lit(new))
                .otherwise(pl.col('language_name'))
                .alias('language_name')
            )
        post_relabel = data['language_name'].unique().len()
        print(f"Reduced number of languages by {post_blacklist - post_relabel} through relabeling (Requested {len([x for y in RELABELED_LANGUAGES.values() for x in y]) - len(RELABELED_LANGUAGES)})")
        # Capitalize
        data = data.with_columns(pl.col('language_name').str.to_titlecase())
        post_titlecase = data['language_name'].unique().len()
        print(f"Converted to titlecase. Reduced number of languages by {post_relabel - post_titlecase}")
        if cache:
            data.write_parquet(CACHE_DIR / 'dataset.parquet')
    else:
        data = pl.read_parquet(CACHE_DIR / 'dataset.parquet')

    return data