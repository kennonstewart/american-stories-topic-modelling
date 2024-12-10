import pandas as pd
import math as math
from datasets import load_dataset
import config

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

def load_dataset_wrapper(phase):
    analysis_start, analysis_end, step_size = getattr(config, f"{phase}_SUBSET_ANALYSIS_START"), getattr(config, f"{phase}_SUBSET_ANALYSIS_END"), getattr(config, f"{phase}_SUBSET_STEP_SIZE")
    print(f"Analysis will be performed on the years {analysis_start} to {analysis_end} in steps of {step_size}")

    subsets = [str(x) for x in range(analysis_start, analysis_end, step_size)]
    dataset = load_dataset("dell-research-harvard/AmericanStories",
                        "subset_years",
                        year_list = subsets,
                        trust_remote_code = True)
    return dataset