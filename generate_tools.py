from utils.data import get_dataset
import polars as pl
import requests

def get_language_info():
    url = f"https://raw.githubusercontent.com/toUpperCase78/tiobe-index-ratings/refs/heads/master/Tiobe_Index_April2025.csv"
    response = requests.get(url)
    df = pl.read_csv(response.content).with_columns(pl.col('Programming Language').str.to_titlecase())
    if 'Fotran' in df['Programming Language'].to_list():
        df = df.with_columns(pl.col('Programming Language').str.replace('Fotran', 'Fortran'))
    if 'Assembly Language' in df['Programming Language'].to_list():
        df = df.with_columns(pl.col('Programming Language').str.replace('Assembly Language', 'Assembly (X86)'))
    return df

if __name__ == "__main__":
    df = get_dataset()
    print(df.head())

    # Get all languages
    languages = df['language_name'].value_counts().sort('count', descending=True)
    for language in languages['language_name'].unique().sort().to_list():
        print(language)

    bruh = get_language_info()
    for language in bruh['Programming Language'].to_list():
        print(f"{language}: {languages.filter(pl.col('language_name') == language)['count'].to_numpy().size}")
