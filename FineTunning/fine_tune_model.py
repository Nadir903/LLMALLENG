from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk, DatasetDict
import torch

# Define the directory where the non-tokenized data is stored
data_dir = '/Users/nadiralcalde/Seminarios/Generative_AI_and_Democracy/LLMALLENG/not_tokenized_dataset'

# Load the tokenizer and model
model_name = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the non-tokenized dataset
dataset = load_from_disk(data_dir)

# Print the dataset structure
print("Dataset Structure:", dataset)

# Verify and print the columns of the dataset
print("Dataset Columns:", dataset.column_names)

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
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
print("Tokenization completed.")

# Ensure the dataset is a DatasetDict and has train/validation splits
if not isinstance(tokenized_dataset, DatasetDict):
    tokenized_dataset = DatasetDict({"train": tokenized_dataset})

# Print the dataset structure
print("Tokenized Dataset Structure:", tokenized_dataset)

# Verify and print the columns of the tokenized dataset
if "train" in tokenized_dataset:
    print("Tokenized Dataset Columns in 'train':", tokenized_dataset["train"].column_names)
else:
    print("Train dataset not found in the tokenized dataset.")

# Use a smaller subset for quick iteration
small_train_dataset = tokenized_dataset["train"].select(range(10000))  # Select only 10,000 examples for training

# Check if MPS device is available and set device accordingly
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Move model to the device
model.to(device)

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
    gradient_accumulation_steps=4,  # accumulate gradients over 4 steps
    use_mps_device=True  # Use MPS device if available
)

# Custom Trainer to move inputs to MPS device
class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        # Move inputs to the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return super().training_step(model, inputs)

# Initialize the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_train_dataset,  # temporarily using train as eval for debugging
    tokenizer=tokenizer
)

# Fine-tune the model
print("Starting training...")
trainer.train()
print("Training completed.")

# Save the model
trainer.save_model("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

# python FineTunning/fine_tune_model.py