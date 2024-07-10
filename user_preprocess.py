import json
from os.path import join, split as split_path, isfile
from datasets import load_dataset
import argparse
import os


def prep_of_files(dataset, source_lang, target_lang):
    translated_v3 = []
    allang_texts = []
    eng_texts = []
    for example in dataset['train']:
        translated_v3.append(
            [0, target_lang, example['translation'][target_lang]]
        )
        allang_texts.append(example['translation'][source_lang])
        eng_texts.append(example['translation'][target_lang])
    return translated_v3, allang_texts, eng_texts


def handle_input_file(file_location, output_path):
    file_name = split_path(file_location)[-1]
    prepro_dosya = f"{file_name}"
    allang_texts_file = join(output_path, 'source_texts.json')
    eng_texts_file = join(output_path, 'target_texts.json')

    if isfile(join(output_path, prepro_dosya)) and isfile(allang_texts_file) and isfile(eng_texts_file):
        print(f"Files already exist. Skipping processing for {file_location}")
        return

    print(f"Processing file: {file_location}")

    # Load datasets
    # Preprocess datasets using the correct keys
    # Combine data into a single list for each language pair

    dataset_es_en = load_dataset("opus100", "en-es")
    dataset_tr_en = load_dataset("opus100", "en-tr")
    dataset_ar_en = load_dataset("opus100", "ar-en")
    translated_es_en, st_es_en, tt_es_en = prep_of_files(dataset_es_en, 'es',
                                                                                               'en')
    translated_tr_en, st_tr_en, tt_tr_en = prep_of_files(dataset_tr_en, 'tr',
                                                                                               'en')
    translated_ar_en, st_ar_en, tt_ar_en = prep_of_files(dataset_ar_en, 'ar',
                                                                                               'en')
    all_comb_trans = translated_es_en + translated_tr_en + translated_ar_en
    all_comb_allang_texts = st_es_en + st_tr_en + st_ar_en
    all_comb_eng_texts = tt_es_en + tt_tr_en + tt_ar_en

    save_to_json(all_comb_trans, all_comb_allang_texts, all_comb_eng_texts, output_path,
                 prepro_dosya, allang_texts_file, eng_texts_file)


def save_to_json(all_comb_trans, all_comb_allang_texts, all_comb_eng_texts, output_path, prepro_dosya_name,
                 allang_texts_file, eng_texts_file):
    preprocessed_data = {
        "transformed_representation": all_comb_trans
    }
    with open(join(output_path, prepro_dosya_name), 'w', encoding='utf-8') as f:
        json.dump(preprocessed_data, f, ensure_ascii=False, indent=4)
    print(f"Success!! Saved prep data : {join(output_path, prepro_dosya_name)}")

    with open(allang_texts_file, 'w', encoding='utf-8') as f:
        json.dump(all_comb_allang_texts, f, ensure_ascii=False, indent=4)
    print(f"Saved Allang Text to {allang_texts_file}")

    with open(eng_texts_file, 'w', encoding='utf-8') as f:
        json.dump(all_comb_eng_texts, f, ensure_ascii=False, indent=4)
    print(f"Saved English Text to {eng_texts_file}")


parser = argparse.ArgumentParser(description='Preprocessing the data.')
parser.add_argument('--input', type=str, help='Input data is here:', required=True, action="append")
parser.add_argument('--output', type=str, help='Output directory is here:', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    files_inp = args.input
    files_out = args.output

    os.makedirs(files_out, exist_ok=True)

    for file_location in files_inp:
        handle_input_file(file_location, files_out)
