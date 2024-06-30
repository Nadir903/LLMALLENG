# This file will be called once the setup is complete. 
# It should perform all the preprocessing steps that are necessary to run the project. 
# All important arguments will be passed via the command line.
# The input files will adhere to the format specified in datastructure/input-file.json

import json
from os.path import join
from datasets import load_dataset
import argparse
import os


# Function to preprocess data
def preprocess_data(dataset, source_lang, target_lang):
    source_texts = []
    target_texts = []
    for example in dataset['train']:
        source_texts.append(example['translation'][source_lang])
        target_texts.append(example['translation'][target_lang])
    return source_texts, target_texts


# Function to handle input file and save preprocessed data
def handle_input_file(file_location, output_path):
    print(f"Processing file: {file_location}")

    # Load datasets
    dataset_es_en = load_dataset("opus100", "en-es")
    dataset_tr_en = load_dataset("opus100", "en-tr")
    dataset_ar_en = load_dataset("opus100", "ar-en")

    # Preprocess datasets using the correct keys
    source_texts_es_en, target_texts_es_en = preprocess_data(dataset_es_en, 'es', 'en')
    source_texts_tr_en, target_texts_tr_en = preprocess_data(dataset_tr_en, 'tr', 'en')
    source_texts_ar_en, target_texts_ar_en = preprocess_data(dataset_ar_en, 'ar', 'en')

    # Combine data into a single list for each language pair
    combined_source_texts = source_texts_es_en + source_texts_tr_en + source_texts_ar_en
    combined_target_texts = target_texts_es_en + target_texts_tr_en + target_texts_ar_en

    # Save preprocessed data to JSON
    save_to_json(combined_source_texts, combined_target_texts, output_path)


# Function to save data to JSON file
def save_to_json(source_texts, target_texts, output_path):
    with open(join(output_path, 'source_texts.json'), 'w', encoding='utf-8') as f:
        json.dump(source_texts, f, ensure_ascii=False, indent=4)
    with open(join(output_path, 'target_texts.json'), 'w', encoding='utf-8') as f:
        json.dump(target_texts, f, ensure_ascii=False, indent=4)
    print(f"Saved source texts to {join(output_path, 'source_texts.json')}")
    print(f"Saved target texts to {join(output_path, 'target_texts.json')}")


parser = argparse.ArgumentParser(description='Preprocess the data.')
parser.add_argument('--input', type=str, help='Path to the input data.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    files_inp = args.input
    files_out = args.output

    # Create output directory if it doesn't exist
    os.makedirs(files_out, exist_ok=True)

    for file_location in files_inp:
        handle_input_file(file_location, files_out)


# to run : python user_preprocess.py --input sample_data/article_1.json --output output

