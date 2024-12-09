import pandas as pd
import time

def test_time_efficiency(articles):
    results = []

    # reset the index of the articles 
    articles = articles.reset_index(drop=True)

    for idx, article in articles.items():
        original_length = len(article) # get the length of the original article

        # define the methods we're using
        methods = {
            "list_comprehension": remove_non_english_list_comprehension,
            "numpy": remove_non_english_numpy,
            "pandas": remove_non_english_pandas,
            "pytorch": remove_non_english_pytorch
        }
        output = {} # initialize the output dictionary
        # define progress as a percentage
        progress = (idx + 1) / len(articles) * 100
        # print the progress    
        print(f"Progress: {progress:.2f}%")
        for name, method in methods.items(): # for every one of the methods proposed
            start_time = time.time() # start the clock
            cleaned_text = method(article, english_words) # clean the text
            processed_article_length = len(cleaned_text) # get the length of the processed article
            elapsed_time = time.time() - start_time # get the runtime
            output[name] = elapsed_time # append that to the output
            print(f"{name} took {elapsed_time:.6f} seconds")
            output["processed_article_length"] = processed_article_length
            print("-" * 50)

        # append the results for that article as a row to the data frame
        results.append({
            'index': idx,
            'original_article_length': original_length,
            'processed_article_length': output["processed_article_length"],
            'list_comprehension_time': output["list_comprehension"],
            'numpy_time': output["numpy"],
            'pandas_time': output["pandas"],
            'pytorch_time': output["pytorch"]
        })

    return pd.DataFrame(results)