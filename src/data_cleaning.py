import config
import pandas as pd

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

  # Convert to pandas DataFrame if needed
  return sampled_dataset