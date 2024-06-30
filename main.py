import subprocess


subprocess.run(['python', 'user_preprocess.py','--input', 'sample_data/article_1.json', '--output', 'output' ])

subprocess.run(['python', 'user_inference.py', '--query', 'bunu cevir', '--query_id','1', '--output', 'output' ])

subprocess.run(['python', 'prepare_dataset_not_tokenized.py'])

subprocess.run(['python', 'fine_tune_model.py'])
