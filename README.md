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
python user_inference.py --query "Nacido el 22 de abril de 1904 en Nueva York, Robert Oppenheimer estudió filosofía, literatura e idiomas (se dice que tenía tanta facilidad para los idiomas que llegó a aprender italiano en un mes). Este hombre polifacético y con múltiples intereses también amaba los clásicos: leía los diálogos de Platón en griego y era un entusiasta del antiguo poema hindú Bhagvad Gita. Oppie, diminutivo por el cual era conocido entre sus allegados, empezó a mostrar interés por la física experimental en la Universidad de Harvard, concretamente mientras cursava la asignatura de termodinámica que impartía el profesor Percy Bridgman." --query_id 1 --output output
python evaluate_model.py 

For translating multiple articles in the sample_data directory use following command: python translator.py --input sample_data
