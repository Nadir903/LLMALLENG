import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
import evaluate
import torch
import json
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the fine-tuned model and tokenizer
model_name = "fine_tuned_model"
logger.info(f"Loading model and tokenizer from {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the test dataset
test_data_path = 'output'  # Path to the directory containing test data (same as output from user_preprocess.py)
source_texts_path = f'{test_data_path}/source_texts.json'
target_texts_path = f'{test_data_path}/target_texts.json'

logger.info(f"Loading test data from {source_texts_path} and {target_texts_path}")
with open(source_texts_path, 'r', encoding='utf-8') as f:
    source_texts = json.load(f)

with open(target_texts_path, 'r', encoding='utf-8') as f:
    target_texts = json.load(f)

# Create a Dataset object for the test data
logger.info("Creating Dataset object for the test data")
test_dataset = Dataset.from_dict({"source_texts": source_texts, "target_texts": target_texts})

# Shuffle and reduce the dataset size to 300 examples
logger.info("Shuffling and reducing dataset size to 300 examples")
test_dataset = test_dataset.shuffle(seed=42).select(range(3000))

# Define the evaluation metric
logger.info("Loading evaluation metric sacrebleu")
metric = evaluate.load("sacrebleu", trust_remote_code=True)

# Tokenize and generate predictions
def evaluate_model(model, tokenizer, dataset):
    logger.info("Starting evaluation...")
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    predictions = []
    references = []

    for i, example in enumerate(dataset):
        logger.info(f"Processing example {i+1}/{len(dataset)}")
        inputs = tokenizer.encode(example['source_texts'], return_tensors="pt", truncation=True, max_length=128).to(device)
        targets = example['target_texts']
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=128)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
        references.append([targets])

    logger.info("Computing metric")
    return metric.compute(predictions=predictions, references=references)

# Run evaluation
logger.info("Running evaluation")
results = evaluate_model(model, tokenizer, test_dataset)
logger.info(f"Evaluation Results: {results}")

print(f"Evaluation Results: {results}")