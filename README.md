# LLMALLENG
LLM all (spanish, turkish, arabish) to english
The whole code is in the master branch, please before making changes, create a branch

1. Recollect and Process Data (opus100) DONE
2. Tokenization and preparation of the DataSet using model DONE
3. FineTuning (training the model) DONE
4. Evaluate and fine-tune the model TODO
5. Save and Share the model TODO


#python user_preprocess.py --input sample_data/article_1.json --output output
#python user_inference.py --query "bunu cevir" --query_id 1 --output output
#python prepare_dataset_not_tokenized.py    #python3 prepare_dataset.py
#python FineTunning/fine_tune_model.py