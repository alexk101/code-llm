from utils.data import get_language_info
from utils.tools import LanguageTools


def verify_language_tools():
    """
    Verify that all languages in the language info are supported by the tools.
    Returns a list of languages that are not supported
    """
    tools = LanguageTools()
    languages = get_language_info()["Programming Language"].to_list()
    print(f"Searching for the following {len(languages)} languages:")
    print(languages)
    missing_languages = []
    present_languages = []
    for language in languages:
        config = tools.get_language_config(language)
        if len(config) == 0:
            missing_languages.append(language)
        else:
            present_languages.append(language)
    if len(missing_languages) > 0:
        print(f"Missing languages: {missing_languages}")
    else:
        print("All languages are supported")
    print(f"Present languages: {present_languages}")
    return missing_languages, present_languages
