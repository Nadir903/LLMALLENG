import argparse
import json
from os.path import join
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def handle_user_query(query, query_id, output_path):
    inputs = tokenizer.encode(query, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = {
        "generated_query": translation,
        "detected_language": "en",
    }

    with open(join(output_path, f"{query_id}.json"), "w") as f:
        json.dump(result, f)


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
