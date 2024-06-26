from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk, DatasetDict

# Define the directory where the preprocessed data is stored
data_dir = '/Users/nadiralcalde/Seminarios/Generative_AI_and_Democracy/LLMALLENG/GatteredData/processed_dataset'

# Load the tokenizer and model
model_name = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the processed dataset
dataset = load_from_disk(data_dir)

# Debugging: Check the dataset columns
print("Dataset Columns:", dataset.column_names)

# Ensure the dataset is a DatasetDict and has train/validation splits
if not isinstance(dataset, DatasetDict):
    dataset = DatasetDict({"train": dataset})

# Use a smaller subset for quick iteration
small_train_dataset = dataset["train"].select(range(10000))  # Select only 10,000 examples for training

# Tokenize the dataset
def preprocess_function(examples):
    inputs = [ex['en'] for ex in examples['translation']]
    targets = [ex['target'] for ex in examples['translation']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing to the dataset
print("Starting tokenization...")
tokenized_dataset = small_train_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
print("Tokenization completed.")

# Debugging: Check the tokenized dataset structure
print("Tokenized Dataset Structure:", tokenized_dataset)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",  # updated key for more frequent evaluation
    eval_steps=100,  # evaluate every 100 steps
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # increased batch size
    per_device_eval_batch_size=16,  # increased batch size
    num_train_epochs=1,  # reduce to 1 epoch for quick iteration
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=500,
    logging_steps=50,  # log every 50 steps
    logging_dir='./logs',
    gradient_accumulation_steps=4  # accumulate gradients over 4 steps
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # temporarily using train as eval for debugging
    tokenizer=tokenizer
)

# Fine-tune the model
print("Starting training...")
trainer.train()
print("Training completed.")

# Save the model
trainer.save_model("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

#pip install transformers[torch] or pip install accelerate -U
#pip install transformers datasets --upgrade
#python FineTunning/fine_tune_model.py
