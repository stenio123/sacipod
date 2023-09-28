from datasets import Dataset

# Saving DataFrame to file
df.to_json("/path/to/dataset.json", orient='records', lines=True)

# Load as HF dataset and push to hub
hf_dataset = Dataset.from_json("/path/to/dataset.json")
hf_dataset.push_to_hub("your_dataset")
