from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_metric, Dataset
import torch
import json

# Load the fine-tuned model and tokenizer
# Change to fine_tuned_model_GPU if necessary
model_name = "fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the test dataset (this should be prepared similarly to your training data)
test_data_path = 'path_to_your_test_data'  # Update with the correct path
source_texts_path = f'{test_data_path}/source_texts.json'
target_texts_path = f'{test_data_path}/target_texts.json'

with open(source_texts_path, 'r', encoding='utf-8') as f:
    source_texts = json.load(f)

with open(target_texts_path, 'r', encoding='utf-8') as f:
    target_texts = json.load(f)

# Create a Dataset object for the test data
test_dataset = Dataset.from_dict({"source_texts": source_texts, "target_texts": target_texts})

# Define the evaluation metric
metric = load_metric("sacrebleu")


# Tokenize and generate predictions
def evaluate(model, tokenizer, dataset):
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    predictions = []
    references = []

    for example in dataset:
        inputs = tokenizer.encode(example['source_texts'], return_tensors="pt", truncation=True, max_length=128).to(
            device)
        targets = example['target_texts']
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=128)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
        references.append([targets])

    return metric.compute(predictions=predictions, references=references)


# Run evaluation
results = evaluate(model, tokenizer, test_dataset)
print(f"Evaluation Results: {results}")

# To run: python evaluate_model.py
