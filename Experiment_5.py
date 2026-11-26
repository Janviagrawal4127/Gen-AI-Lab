
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

model_name = "t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("samsum")

train_dataset = dataset["train"].select(range(1000))
eval_dataset = dataset["validation"].select(range(200))

def preprocess_function(examples):
    inputs = ["summarize: " + dialogue for dialogue in examples["dialogue"]]

    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True
    )

    labels = tokenizer(
        text_target=examples["summary"],
        max_length=128,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_eval = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-summarization",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    num_train_epochs=2,
    predict_with_generate=True,
    save_total_limit=1,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    processing_class=tokenizer
)

trainer.train()

model.save_pretrained("./fine_tuned_t5")
tokenizer.save_pretrained("./fine_tuned_t5")

text = """
Alice: Hey, are we meeting for the project today?
Bob: Yes, let's meet at 4 PM in the library.
Alice: Great! I will bring the project report.
Bob: Perfect, see you there.
"""

inputs = tokenizer(
    "summarize: " + text,
    return_tensors="pt",
    max_length=512,
    truncation=True
)

device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=100,
    min_length=10,
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("Original Text:")
print(text)

print("\nGenerated Summary:")
print(summary)

print("\nModel fine-tuning completed successfully!")

