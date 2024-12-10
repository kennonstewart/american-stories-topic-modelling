import config
import pandas as pd
from cleaning_functions import remove_non_english_list_comprehension
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("averaged_perceptron_tagger_eng")
nltk.download("wordnet")


def subsample_hf_dataset(ds, n):
  seed_value = config.SUBSAMPLE_SEED_VALUE
  print(f"subsampling dataset with seed value {seed_value}")
  # Sample n observations per year
  sampled_data = []
  sample_size = config.SUBSAMPLE_SAMPLE_SIZE
  subsets = [str(x) for x in ds.keys()]
  for year in subsets:
      yearly_data = ds[year]
      # Sample n observations (or all if less than n)
      sample_size = min(n, len(yearly_data))
      print(f"sample size of year {year}: {sample_size}")
      sampled_yearly_data = yearly_data.shuffle(seed=seed_value).select(range(sample_size))
      sampled_data.append(sampled_yearly_data)

  # Concatenate all sampled yearly data into one dataset
  sampled_dataset = pd.concat([d.to_pandas() for d in sampled_data], ignore_index=True)

  # clean article text
  sampled_dataset = clean_article_text(sampled_dataset)

  # clean the non-english article words
  print(f"subsample hf datasets: shape of dataset before cleaning: {sampled_dataset.shape}")
  sampled_dataset["article"] = perform_list_comp_cleaning(sampled_dataset)["article"]

  # Convert to pandas DataFrame if needed
  return sampled_dataset

def cleaning_article_text(article_text):
    # set case to lower
    article_text = article_text.lower()

    # remove punctuation using regex
    article_text = re.sub(r'[^\w\s]', '', article_text)

    # strip out new-line delimiters
    article_text = article_text.replace("\n", " ")

    # remove non-ascii characters
    article_text = article_text.encode("ascii", "ignore").decode()

    # lemmatize the passage
    article_text = lemmatize_passage(article_text)

    return article_text

def clean_article_text(dataset):
    output = dataset
    output["article"] = dataset["article"].apply(cleaning_article_text)
    return output

def perform_list_comp_cleaning(dataset):
   english_words = set(nltk.corpus.words.words())
   dataset["article"] = dataset["article"].apply(lambda x: remove_non_english_list_comprehension(x, english_words))
   print(f"perform list comprehension cleaning: shape of dataset after cleaning: {dataset.shape}")
   return dataset

def isolate_interesting_sources(dataset):
    # get the list of sources
    interesting_sources = config.INTERESTING_SOURCES

    # make a list of yearly dataFrames
    yearly_data = [pd.DataFrame(data) for data in dataset.values()]

    # concatenate the dataFrames
    output = pd.concat(yearly_data)

    # filter the data for interesting sources
    output = output[output["newspaper_name"].isin(interesting_sources)]

    # remove punctuation and stop words
    output = clean_article_text(output)

    # remove non-English words
    output = perform_list_comp_cleaning(output)

    return output

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:         
        return wordnet.NOUN
       
def lemmatize_passage(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    lemmatized_sentence = ' '.join(lemmatized_words)
    return lemmatized_sentence