import json  # json kütüphanesini ekledik
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset


model_name = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the processed data from user_preprocess.py
processed_data_path = 'output'  # Assuming output directory from user_preprocess.py
source_texts_path = f'{processed_data_path}/source_texts.json'
target_texts_path = f'{processed_data_path}/target_texts.json'

with open(source_texts_path, 'r', encoding='utf-8') as f:
    source_texts = json.load(f)

with open(target_texts_path, 'r', encoding='utf-8') as f:
    target_texts = json.load(f)

# Create Dataset objects using datasets library
datasets = DatasetDict({
    "train": Dataset.from_dict({"source_texts": source_texts, "target_texts": target_texts})
})

# Tokenize the dataset
def preprocess_function(examples):
    inputs = examples['source_texts']
    targets = examples['target_texts']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing to the dataset
print("Starting tokenization...")
tokenized_dataset = datasets["train"].map(preprocess_function, batched=True, remove_columns=["source_texts", "target_texts"])
print("Tokenization completed.")

# Debugging: Check the tokenized dataset structure
print("Tokenized Dataset Structure:", tokenized_dataset)

# Save the tokenized dataset (optional)
tokenized_dataset.save_to_disk("tokenized_dataset")

# To run: python3 prepare_dataset.py
