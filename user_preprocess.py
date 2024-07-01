import json
from os.path import join, split as split_path, isfile
from datasets import load_dataset
import argparse
import os


# Function to preprocess data
def preprocess_data(dataset, source_lang, target_lang):
    transformed_representation = []
    source_texts = []
    target_texts = []
    for example in dataset['train']:
        transformed_representation.append(
            [0, target_lang, example['translation'][target_lang]]
        )
        source_texts.append(example['translation'][source_lang])
        target_texts.append(example['translation'][target_lang])
    return transformed_representation, source_texts, target_texts


# Function to handle input file and save preprocessed data
def handle_input_file(file_location, output_path):
    file_name = split_path(file_location)[-1]
    preprocessed_file_name = f"{file_name}"
    source_texts_file = join(output_path, 'source_texts.json')
    target_texts_file = join(output_path, 'target_texts.json')

    if isfile(join(output_path, preprocessed_file_name)) and isfile(source_texts_file) and isfile(target_texts_file):
        print(f"Files already exist. Skipping processing for {file_location}")
        return

    print(f"Processing file: {file_location}")

    # Load datasets
    dataset_es_en = load_dataset("opus100", "en-es")
    dataset_tr_en = load_dataset("opus100", "en-tr")
    dataset_ar_en = load_dataset("opus100", "ar-en")

    # Preprocess datasets using the correct keys
    transformed_representation_es_en, source_texts_es_en, target_texts_es_en = preprocess_data(dataset_es_en, 'es',
                                                                                               'en')
    transformed_representation_tr_en, source_texts_tr_en, target_texts_tr_en = preprocess_data(dataset_tr_en, 'tr',
                                                                                               'en')
    transformed_representation_ar_en, source_texts_ar_en, target_texts_ar_en = preprocess_data(dataset_ar_en, 'ar',
                                                                                               'en')

    # Combine data into a single list for each language pair
    combined_transformed_representation = transformed_representation_es_en + transformed_representation_tr_en + transformed_representation_ar_en
    combined_source_texts = source_texts_es_en + source_texts_tr_en + source_texts_ar_en
    combined_target_texts = target_texts_es_en + target_texts_tr_en + target_texts_ar_en

    # Save preprocessed data to JSON
    save_to_json(combined_transformed_representation, combined_source_texts, combined_target_texts, output_path,
                 preprocessed_file_name, source_texts_file, target_texts_file)


# Function to save data to JSON file
def save_to_json(transformed_representation, source_texts, target_texts, output_path, preprocessed_file_name,
                 source_texts_file, target_texts_file):
    preprocessed_data = {
        "transformed_representation": transformed_representation
    }
    with open(join(output_path, preprocessed_file_name), 'w', encoding='utf-8') as f:
        json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)
    print(f"Saved preprocessed data to {join(output_path, preprocessed_file_name)}")

    with open(source_texts_file, 'w', encoding='utf-8') as f:
        json.dump(source_texts, f, ensure_ascii=False, indent=4)
    print(f"Saved source texts to {source_texts_file}")

    with open(target_texts_file, 'w', encoding='utf-8') as f:
        json.dump(target_texts, f, ensure_ascii=False, indent=4)
    print(f"Saved target texts to {target_texts_file}")


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

# to run: python user_preprocess.py --input sample_data/article_1.json --output output
