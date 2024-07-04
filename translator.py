import json
import argparse
import os
from os.path import join, isfile, basename
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
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


def translate_article(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        article = json.load(f)

    content = article["content"]
    query_id = article["timestamp"]  # Use timestamp as query_id to ensure uniqueness

    handle_user_query(content, query_id, output_dir)

    with open(join(output_dir, f"{query_id}.json"), 'r', encoding='utf-8') as f:
        result = json.load(f)

    article["translated_content"] = result["generated_query"]

    output_file = join(output_dir, f"translated_{basename(input_file)}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(article, f, ensure_ascii=False, indent=4)
    print(f"Translated article saved to {output_file}")


def translate_articles_in_directory(input_dir):
    output_dir = 'translated_articles'

    if not os.path.isdir(input_dir):
        print(f"The input path {input_dir} is not a directory.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        input_file = join(input_dir, file_name)
        if isfile(input_file) and file_name.endswith('.json'):
            print(f"Translating {input_file}...")
            translate_article(input_file, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate the content of all JSON articles in a directory.')
    parser.add_argument('--input_dir', type=str, help='Path to the input directory containing JSON files.',
                        required=True)

    args = parser.parse_args()
    input_dir = args.input_dir

    translate_articles_in_directory(input_dir)

# python translator.py --input sample_data
