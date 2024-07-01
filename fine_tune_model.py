from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk, DatasetDict
import torch

# Define the directory where the non-tokenized data is stored
data_dir = 'not_tokenized_dataset'

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
        labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
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

# Explicitly set the device to CPU
device = torch.device("cpu")
print("Using CPU device")

# Move model to the device
model.to(device)


# Define a function to explicitly move internal tensors to the CPU
def move_tensors_to_device(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(move_tensors_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return {k: move_tensors_to_device(v, device) for k, v in tensor.items()}
    else:
        return tensor


# Custom Trainer to move inputs to CPU device
class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        # Move inputs to the correct device
        inputs = move_tensors_to_device(inputs, device)
        model = model.to(device)
        return super().training_step(model, inputs)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",  # updated key for more frequent evaluation
    eval_steps=100,  # evaluate every 100 steps
    learning_rate=2e-5,
    per_device_train_batch_size=8,  #use only to make possible to run
    per_device_eval_batch_size=8,
    num_train_epochs=1,  # reduce to 1 epoch for quick iteration
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=500,
    logging_steps=50,  # log every 50 steps
    logging_dir='./logs',
    gradient_accumulation_steps=4,  # accumulate gradients over 4 steps
    use_mps_device=False  # Ensure not using MPS device
)


# Ensure internal states in the forward pass are on CPU device
class CustomModel(AutoModelForSeq2SeqLM):
    def forward(self, *args, **kwargs):
        # Move all inputs and kwargs to the correct device
        args = tuple(move_tensors_to_device(arg, device) for arg in args)
        kwargs = {k: move_tensors_to_device(v, device) for k, v in kwargs.items()}
        return super().forward(*args, **kwargs)


# Initialize the model with the custom class
model = CustomModel.from_pretrained(model_name).to(device)

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

# to run : python fine_tune_model.py
