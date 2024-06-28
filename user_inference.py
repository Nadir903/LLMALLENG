# This file will be executed when a user wants to query your project.
import argparse
from os.path import join
import json
import requests as r
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Done Implement the inference logic here
#Tokenizer

# Load the tokenizer and model :pip install sentencepiece
model_name = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Function to handle user query
def handle_user_query(query, query_id, output_path):
    inputs = tokenizer.encode(query, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = {
        "generated_queries": [translation],
        "detected_language": "en",
    }

    with open(join(output_path, f"{query_id}.json"), "w") as f:
        json.dump(result, f)


# This is a sample argparse-setup, you probably want to use in your project:
parser = argparse.ArgumentParser(description='Run the inference.')
parser.add_argument('--query', type=str, help='The user query.', required=True, action="append")
parser.add_argument('--query_id', type=str, help='The IDs for the queries, in the same order as the queries.',
                    required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    queries = args.query
    query_ids = args.query_id
    output = args.output

    assert len(queries) == len(query_ids), "The number of queries and query IDs must be the same."

    for query, query_id in zip(queries, query_ids):
        handle_user_query(query, query_id, output)

#Translation Step

#python user_inference.py --query "Translate this text" --query_id 1 --output /Users/nadiralcalde/Seminarios/Generative_AI_and_Democracy/LLMALLENG/output
