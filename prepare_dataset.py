# fine_tune_model.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

# Load the tokenizer and model
model_name = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the processed dataset
dataset = load_from_disk('GatteredData/processed_dataset')

# Tokenize the dataset
def preprocess_function(examples):
    inputs = [ex['en'] for ex in examples['translation']]
    targets = [ex['target'] for ex in examples['translation']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=500
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

# python FineTunning/fine_tune_model.py
