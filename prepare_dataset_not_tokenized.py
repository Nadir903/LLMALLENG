# prepare_dataset_not_tokenized.py
import json
from datasets import Dataset
import os


def load_data(source_file, target_file):
    with open(source_file, 'r', encoding='utf-8') as f:
        source_texts = json.load(f)
    with open(target_file, 'r', encoding='utf-8') as f:
        target_texts = json.load(f)
    return source_texts, target_texts


def create_dataset(source_texts, target_texts):
    data = {'translation': [{'en': src, 'target': tgt} for src, tgt in zip(source_texts, target_texts)]}
    dataset = Dataset.from_dict(data)
    return dataset


output_dir = 'output'
source_texts_file = os.path.join(output_dir, 'source_texts.json')
target_texts_file = os.path.join(output_dir, 'target_texts.json')

if not os.path.exists(source_texts_file) or not os.path.exists(target_texts_file):
    print(f"One or both files not found: {source_texts_file}, {target_texts_file}")
else:
    source_texts, target_texts = load_data(source_texts_file, target_texts_file)
    dataset = create_dataset(source_texts, target_texts)
    dataset.save_to_disk('not_tokenized_dataset')
    print("Dataset saved to 'not_tokenized_dataset'")
