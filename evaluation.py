from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

# Load tokenizer and model
model_name = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load evaluation dataset
dataset = load_dataset("json", data_files={"validation": "validation.json"}, field="data")
validation_dataset = dataset["validation"]

# Define metric for evaluation (e.g., BLEU)
metric = load_metric("sacrebleu")

# Function to compute metrics during evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate BLEU score
    bleu_score = metric.compute(predictions=decoded_preds, references=[decoded_labels])
    return {"bleu": bleu_score["score"]}

# Training arguments
training_args = TrainingArguments(
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=3,
    output_dir="./results",
    overwrite_output_dir=True,
)

# Trainer for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    train_dataset=None,  # Not training, only evaluating
    eval_dataset=validation_dataset,
)

# Evaluate the model
evaluation_results = trainer.evaluate()

# Print evaluation results
print("Evaluation results:", evaluation_results)

# Check if fine-tuning is necessary based on evaluation metrics (e.g., BLEU score)
if evaluation_results["bleu"] < 30.0:  # Adjust threshold based on your requirements
    print("Model performance is below threshold. Fine-tuning is recommended.")

    # Define fine-tuning arguments
    fine_tuning_args = TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=3,
        output_dir="./fine_tuned_results",
        overwrite_output_dir=True,
        num_train_epochs=1,  # Example: Fine-tune for 1 epoch
    )

    # Fine-tune the model
    trainer_fine_tune = Trainer(
        model=model,
        args=fine_tuning_args,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        train_dataset=train_dataset,  # Provide your training dataset here
        eval_dataset=validation_dataset,
    )

    print("Starting fine-tuning...")
    trainer_fine_tune.train()
    print("Fine-tuning completed.")

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
else:
    print("Model performance is satisfactory. No fine-tuning needed.")