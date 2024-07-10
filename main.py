import subprocess

# Step 1: Recollect and Process Data using a large dataset (opus100)
subprocess.run(['python', 'user_preprocess.py', '--input', 'sample_data/article_1.json', '--output', 'output'])

# Step 2: Preparation of the DataSet
subprocess.run(['python', 'prepare_dataset_not_tokenized.py'])

# Step 3: Tokenization and FineTuning (training the model)
# Note: Modify the command based on your hardware (CPU or GPU)
# For CPU usage
subprocess.run(['python', 'fine_tune_model.py'])
# For GPU usage
# subprocess.run(['python', 'fine_tune_model_GPU.py'])

# Step 4: Evaluation of the model and its accuracy
subprocess.run(['python', 'evaluate_model.py'])

# Step 5: Translation of simple sentences
subprocess.run(['python', 'user_inference.py', '--query', 'Here comes your query in any source language', '--query_id', '1', '--output', 'output'])

# Step 6: Translation of a set of articles
subprocess.run(['python', 'translator.py', '--input', 'sample_data'])

# Step 7: Test evaluation of all previous steps Test specific parts by changing the argument: {preprocess, setup,
# inference, prepare_dataset, fine_tune, evaluate, translate}
subprocess.run(['python', 'test.py', '--part', 'translate'])  # Change 'translate' to the part you want to test
