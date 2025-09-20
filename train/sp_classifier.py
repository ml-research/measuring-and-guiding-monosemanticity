
# from huggingface_hub import notebook_login
# notebook_login()

# Is needed for the Trainer
# git clone https://github.com/NVIDIA/apex
# cd apex
# python setup.py install

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import pipeline
import torch



sp_ds = load_from_disk(
    "./datasets/Shakespeare"
)
sp_ds = sp_ds.map(lambda x: {"label": 0 if x["label"] == "modern" else 1})
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_sp_ds = sp_ds.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
recall = evaluate.load("recall")
precision = evaluate.load("precision")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"acc": accuracy.compute(
        predictions=predictions, references=labels
    ), "f1": f1.compute(predictions=predictions, references=labels), "recall": recall.compute(predictions=predictions, references=labels), "precision": precision.compute(predictions=predictions, references=labels)}


id2label = {0: "modern", 1: "shakespearean"}
label2id = {"modern": 0, "shakespearean": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "./llama3_SAE/Shakespeare_Classifier/checkpoint-4600",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)


training_args = TrainingArguments(
    output_dir="Shakespeare_Classifier_eval",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_sp_ds["train"],
    eval_dataset=tokenized_sp_ds["valid"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()
trainer.evaluate(eval_dataset=tokenized_sp_ds["valid"])


