# LLMALLENG — Multilingual-to-English Neural Translation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface&logoColor=white)
![Helsinki-NLP](https://img.shields.io/badge/Model-Helsinki--NLP-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Fine-tuned neural machine translation pipeline that translates **Spanish, Turkish, and Arabic** into English, built on top of the [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) MarianMT models and the [opus100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) multilingual dataset.

Developed as part of the *Science Seminar: Generative AI and Democracy* at the **Technical University of Munich (TUM)**, 2024.

---

## What it does

- Preprocesses and tokenizes raw multilingual text corpora (opus100)
- Fine-tunes a pre-trained MarianMT model on sentence-pair data
- Evaluates translation quality on a held-out test set
- Runs inference on arbitrary input text or JSON article files
- Supports CPU and GPU training

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Models | Helsinki-NLP / MarianMT (HuggingFace) |
| Training data | opus100 (Spanish, Turkish, Arabic → English) |
| Libraries | `transformers`, `datasets`, `torch`, `sacrebleu` |

---

## Project structure

```
LLMALLENG/
├── user_preprocess.py              # Preprocess raw input articles
├── prepare_dataset_not_tokenized.py # Build sentence-pair dataset
├── fine_tune_model.py              # Fine-tune on CPU
├── fine_tune_model_GPU.py          # Fine-tune on GPU
├── evaluate_model.py               # Evaluate translation quality
├── user_inference.py               # Run inference on a query
├── translator.py                   # Batch translation of articles
├── test.py                         # End-to-end test suite
└── sample_data/                    # Example input files
```

---

## Setup

```bash
git clone https://github.com/Nadir903/LLMALLENG.git
cd LLMALLENG
pip install transformers datasets torch sacrebleu
```

---

## Usage

Run the steps in order:

```bash
# 1. Preprocess raw input
python user_preprocess.py --input sample_data/article_1.json --output output

# 2. Prepare the sentence-pair dataset
python prepare_dataset_not_tokenized.py

# 3. Fine-tune (choose CPU or GPU)
python fine_tune_model.py
python fine_tune_model_GPU.py   # if CUDA is available

# 4. Evaluate translation quality
python evaluate_model.py

# 5. Run inference on a single query
python user_inference.py --query "Your text in any source language" --query_id 1 --output output

# 6. Batch-translate articles
python translator.py --input sample_data

# 7. Run the full test suite
python test.py --part {preprocess,setup,inference,prepare_dataset,fine_tune,evaluate,translate}
```

---

## Background

This project was developed during the *Generative AI and Democracy* science seminar at TUM (Summer 2024), which focused on the intersection of large language models and democratic discourse. The translation pipeline was designed to make multilingual political and news content accessible in English for cross-lingual analysis.

---

## Authors

Nadir Williams Alcalde Echegaray · Ezgi Hasret Açıkgöz · Mohamed Bouhali  
TU München, 2024
