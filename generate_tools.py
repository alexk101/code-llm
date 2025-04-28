from utils.data import get_dataset

if __name__ == "__main__":
    df = get_dataset()
    print(df.head())

    # Get all languages
    languages = df['language_name'].unique().sort()
    for language in languages:
        print(language)
