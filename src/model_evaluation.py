from nltk.tokenize import word_tokenize
import gensim
from gensim.models.coherencemodel import CoherenceModel

def evaluate_model(model, data):
    # Get the topics from the model
    topics = model.get_topics()

    # Convert topics to the required format
    formatted_topics = [[word for word, _ in topic] for topic in topics.values()]

    # Assuming you have a list of documents, where each document is a list of words
    documents = [
        word_tokenize(doc.lower()) for doc in data["article"]
    ]

    # Create a dictionary and corpus required for coherence model
    dictionary = gensim.corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Compute coherence score using 'c_v' coherence measure
    coherence_model = CoherenceModel(topics=formatted_topics, texts=documents, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    return coherence_score

def score_models(models_list, articles):
    scores = []
    for idx in range(len(models_list)):
        score = evaluate_model(models_list[idx], articles[idx])
        scores.append(score)
    return scores