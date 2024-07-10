from transformers import AutoModelForSeq2SeqLM as autoModel
from transformers import AutoTokenizer as tokenizador
from transformers import Trainer as entrenador
from transformers import TrainingArguments as arguments


from datasets import load_from_disk as disk
from datasets import DatasetDict as set
import torch

data_dir = 'not_tokenized_dataset'

model_name = "Helsinki-NLP/opus-mt-mul-en"
mi_tokenizer = tokenizador.from_pretrained(model_name)
mi_model = autoModel.from_pretrained(model_name)

dataset = disk(data_dir)

print("Dataset Structure:", dataset)
print("Dataset Columns:", dataset.column_names)


# This is a tokenization function
def preprocess_function(examples):
    inputs = [ex['en'] for ex in examples['translation']]
    targets = [ex['target'] for ex in examples['translation']]
    model_inputs = mi_tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with mi_tokenizer.as_target_tokenizer():
        labels = mi_tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Starting tokenization...")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
print("Tokenization completed.")

# This ensures the data set is in the correct format
if not isinstance(tokenized_dataset, set):
    tokenized_dataset = set({"train": tokenized_dataset})

print("Tokenized Dataset Structure:", tokenized_dataset)

if "train" in tokenized_dataset:
    print("Tokenized Dataset Columns in 'train':", tokenized_dataset["train"].column_names)
else:
    print("Train dataset not found in the tokenized dataset.")

# select subset due to hardware limitations. The greater the dataset is, the more accuracy the model has
small_train_dataset = tokenized_dataset["train"].select(range(1000))

device = torch.device("cuda")
print("Using GPU device")
mi_model.to(device)


# Tensors form library Torch : multidimensional arrays are moved to the cpu
def move_tensores_to_dispositivos(tensor, device):
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(move_tensores_to_dispositivos(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return {k: move_tensores_to_dispositivos(v, device) for k, v in tensor.items()}
    else:
        return tensor


class CustomTrainer(entrenador):
    def training_step(self, model, inputs):
        inputs = move_tensores_to_dispositivos(inputs, device)
        model = model.to(device)
        return super().training_step(model, inputs)


training_args = arguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=500,
    logging_steps=50,
    logging_dir='./logs',
    gradient_accumulation_steps=4,
    use_mps_device=False
)


# Inputs are moved to correct device during the forward pass.
class CustomModel(autoModel):
    def forward(self, *args, **kwargs):
        args = tuple(move_tensores_to_dispositivos(arg, device) for arg in args)
        kwargs = {k: move_tensores_to_dispositivos(v, device) for k, v in kwargs.items()}
        return super().forward(*args, **kwargs)


model = CustomModel.from_pretrained(model_name).to(device)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_train_dataset,
    tokenizer=mi_tokenizer
)

print("Starting training...")
trainer.train()
print("Training completed.")

trainer.save_model("fine_tuned_model")
mi_tokenizer.save_pretrained("fine_tuned_model")
