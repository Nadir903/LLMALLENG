# This file will be called once the setup is complete. 
# It should perform all the preprocessing steps that are necessary to run the project. 
# All important arguments will be passed via the command line.
# The input files will adhere to the format specified in datastructure/input-file.json

import json
from os.path import join, split as split_path
from datasets import load_dataset
# This is a useful argparse-setup, you probably want to use in your project:
import argparse


# Done Implement the preprocessing steps here

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
    # Load datasets
    dataset_es_en = load_dataset("opus100", "en-es")
    dataset_tr_en = load_dataset("opus100", "en-tr")
    dataset_ar_en = load_dataset("opus100", "ar-en")

    # Preprocess datasets using the correct keys
    source_texts_es_en, target_texts_es_en = preprocess_data(dataset_es_en, 'en', 'es')
    source_texts_tr_en, target_texts_tr_en = preprocess_data(dataset_tr_en, 'en', 'tr')
    source_texts_ar_en, target_texts_ar_en = preprocess_data(dataset_ar_en, 'en', 'ar')

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

    for file_location in files_inp:
        handle_input_file(file_location, files_out)


#python user_preprocess.py --input /Users/nadiralcalde/Seminarios/Generative_AI_and_Democracy/LLMALLENG/sample_data/article_1.json --output /Users/nadiralcalde/Seminarios/Generative_AI_and_Democracy/LLMALLENG/Preproced

