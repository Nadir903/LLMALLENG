# LLMALLENG
LLM all (spanish, turkish, arabish) to english
The whole code is in the master branch, please before making changes, create a branch

1. Recollect and Process Data (opus100) 
2. Preparation of the DataSet using model 
3. Tokenization and FineTuning (training the model) 
4. Translation of input 
5. Evaluation of the model 

This a sequence of command to execute all task. Please follow them in the order they are to be found: 

python user_preprocess.py --input sample_data/article_1.json --output output
python prepare_dataset_not_tokenized.py  
python fine_tune_model.py
python user_inference.py --query "your input in es, ar, tur" --query_id 1 --output output
python evaluate_model.py 
