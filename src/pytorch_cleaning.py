import torch
from nltk.tokenize import word_tokenize

# Load English words and create a vocabulary
def generate_pytorch_mappings(english_words):
    word_to_idx = {word: idx for idx, word in enumerate(english_words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word

# PyTorch-based method
def remove_non_english_pytorch(text, english_words):
    tokens = word_tokenize(text)  # Tokenize text
    word_to_idx, idx_to_word = generate_pytorch_mappings(english_words)
    token_ids = torch.tensor(
        [word_to_idx.get(word.lower(), -1) for word in tokens], dtype=torch.int32
    )  # Map words to IDs (-1 if not in vocabulary)

    mask = token_ids >= 0  # Mask for valid English words
    filtered_tokens = [tokens[i] for i in torch.nonzero(mask).squeeze(1)]  # Filter tokens
    return ' '.join(filtered_tokens)