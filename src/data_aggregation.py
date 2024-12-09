import pandas as pd
import math as math


def yearly_word_counts(ds):
    # list to hold the dataFrames
    dataframes = list()

    # Iterate over each year in the dataset
    for year, data in ds.items():
        print(f"Number of articles in {year}: {len(data['article'])}")
        
        # Calculate word counts for each article
        word_counts = [(year, len(article.split())) for article in data['article']]
        word_counts = pd.DataFrame(data = word_counts, columns = ["year", "word_count"])
        word_counts["log_word_count"] = word_counts["word_count"].apply(math.log)

        # append the DataFrame to the list
        dataframes.append(word_counts)

    # concatenate the DataFrames
    output = pd.concat(dataframes)

    return output