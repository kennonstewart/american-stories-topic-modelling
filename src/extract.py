from datasets import load_dataset
import config

def load_dataset_wrapper(phase):
    analysis_start, analysis_end, step_size = getattr(config, f"{phase}_SUBSET_ANALYSIS_START"), getattr(config, f"{phase}_SUBSET_ANALYSIS_END"), getattr(config, f"{phase}_SUBSET_STEP_SIZE")
    print(f"Analysis will be performed on the years {analysis_start} to {analysis_end} in steps of {step_size}")

    subsets = [str(x) for x in range(analysis_start, analysis_end, step_size)]
    dataset = load_dataset("dell-research-harvard/AmericanStories",
                        "subset_years",
                        year_list = subsets,
                        trust_remote_code = True)
    return dataset