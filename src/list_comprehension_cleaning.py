from nltk.tokenize import word_tokenize

def remove_non_english_list_comprehension(text, english_words):
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word.lower() in english_words])