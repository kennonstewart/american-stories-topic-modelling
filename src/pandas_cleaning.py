import pandas as pd
from nltk.tokenize import word_tokenize

def remove_non_english_pandas(text, english_words):
    tokens = pd.Series(word_tokenize(text))
    mask = tokens.str.lower().isin(english_words)
    return ' '.join(tokens[mask])