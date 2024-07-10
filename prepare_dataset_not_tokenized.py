import json
from datasets import Dataset
import os


def source_target_data(all_file, en_file):
    with open(all_file, 'r', encoding='utf-8') as f:
        alllang_texts = json.load(f)
    with open(en_file, 'r', encoding='utf-8') as f:
        eng_texts = json.load(f)
    return alllang_texts, eng_texts


# Format compatible with Hugging Face dataset library
def dataset_creation(alllang_texts, eng_texts):
    data = {'translation': [{'en': src, 'target': tgt} for src, tgt in zip(alllang_texts, eng_texts)]}
    dataset = Dataset.from_dict(data)
    return dataset


output_dir = 'output'
es_tr_ar_file = os.path.join(output_dir, 'source_texts.json')
en_file = os.path.join(output_dir, 'target_texts.json')

if not os.path.exists(es_tr_ar_file) or not os.path.exists(en_file):
    print(f"One or both files not found: {es_tr_ar_file}, {en_file}")
else:
    alllang_texts, eng_texts = source_target_data(es_tr_ar_file, en_file)
    dataset = dataset_creation(alllang_texts, eng_texts)
    # this saves the data in Apache Arrow format (fast access and memory efficient)
    dataset.save_to_disk('not_tokenized_dataset')
    print("Dataset saved to 'not_tokenized_dataset'")
