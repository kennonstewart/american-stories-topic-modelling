import numpy as np

def remove_non_english_numpy(text, english_words):
    tokens = np.array(word_tokenize(text))
    mask = np.vectorize(lambda word: word.lower() in english_words)(tokens)
    return ' '.join(tokens[mask])