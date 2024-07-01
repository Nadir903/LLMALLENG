import subprocess


subprocess.run(['python', 'user_preprocess.py','--input', 'sample_data/article_1.json', '--output', 'output' ])

subprocess.run(['python', 'prepare_dataset_not_tokenized.py'])

subprocess.run(['python', 'fine_tune_model.py'])

subprocess.run(['python', 'evaluate_model.py'])

subprocess.run(['python', 'user_inference.py', '--query', 'your input in es, ar, tur', '--query_id','1', '--output', 'output' ])


