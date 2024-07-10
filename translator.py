import json
import argparse
import os
from os.path import join as juntador
from os.path import isfile as archivo_valido
from os.path import basename as la_base

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


n = "fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(n)
model = AutoModelForSeq2SeqLM.from_pretrained(n)


def handle_user_query(query, query_id, output_path):
    inputs = tokenizer.encode(query, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = {
        "generated_query": translation,
        "detected_language": "en",
    }

    with open(juntador(output_path, f"{query_id}.json"), "w") as f:
        json.dump(result, f)


def translate_article(input_file, output_dir):
    with open(input_file, 'r', encoding='utf-8') as f:
        article = json.load(f)

    content = article["content"]
    query_id = article["timestamp"]

    handle_user_query(content, query_id, output_dir)

    with open(juntador(output_dir, f"{query_id}.json"), 'r', encoding='utf-8') as f:
        result = json.load(f)

    article["translated_content"] = result["generated_query"]

    o = juntador(output_dir, f"translated_{la_base(input_file)}")
    with open(o, 'w', encoding='utf-8') as f:
        json.dump(article, f, ensure_ascii=False, indent=4)
    print(f"Translated article saved to {o}")


def translate_articles_in_directory(i_dir):
    o_dir = 'translated_articles'

    if not os.path.isdir(i_dir):
        print(f"{i_dir}  not directory.")
        return

    os.makedirs(o_dir, exist_ok=True)

    for f in os.listdir(i_dir):
        input_file = juntador(i_dir, f)
        if archivo_valido(input_file) and f.endswith('.json'):
            print(f"Translating {input_file}...")
            translate_article(input_file, o_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='All articles translated.' )
    parser.add_argument('--input_dir', type=str, help='Path to directory with JSON files.',
                        required=True)

    args = parser.parse_args()
    i_dir = args.input_dir

    translate_articles_in_directory(i_dir)

# python translator.py --input sample_data
