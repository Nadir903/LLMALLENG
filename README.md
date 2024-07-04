# LLMALLENG
LLM all languages all to english
Welcome to our repository. This implementation is translation model where the source languages spanish, turkish as well as arabic and the target language is english.  
In order to set up this LLM, please follow these step.

1. Recollect and Process Data using large data set (opus100). 
2. Preparation of the DataSet mapping the sentences in the source language to its peer in the target language.  
3. Tokenization and FineTuning (training the model). 
4. Evaluation of the model and its accuracy.
5. Translation of simple sentences.
6. Translation of a set of articles.
7. Test evaluation of all previous steps.

This a sequence of command to execute all task. Please follow them in the order they are to be found: 

1. python user_preprocess.py --input sample_data/article_1.json --output output
2. python prepare_dataset_not_tokenized.py  
3. python fine_tune_model.py
4. python evaluate_model.py 
5. python user_inference.py --query "Here comes your query in any source language" --query_id 1 --output output
6. python translator.py --input sample_data
7.  python test.py --part translate argument

For testing, please choose any of this argument which you want to test: {preprocess,setup,inference,prepare_dataset,fine_tune,evaluate,translate}

