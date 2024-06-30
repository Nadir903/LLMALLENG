# prepare_dataset.py
import json
from datasets import Dataset


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


source_texts_file = 'output/source_texts.json'
target_texts_file = 'output/target_texts.json'

source_texts, target_texts = load_data(source_texts_file, target_texts_file)
dataset = create_dataset(source_texts, target_texts)

dataset.save_to_disk('not_tokenized_dataset')

# to run : python prepare_dataset_not_tokenized.py

