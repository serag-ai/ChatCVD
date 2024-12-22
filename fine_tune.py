import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import evaluate
import wandb
import numpy as np
from huggingface_hub import login
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# Set up WandB and Hugging Face
wandb.login(key="your_wandb_key")
login("your_huggingface_token")

# Load and prepare data
def load_data(train_path, val_path, test_path):
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    # Encode labels
    label_encoder = LabelEncoder()
    all_labels = pd.concat([df_train['CVD_Risk'], df_val['CVD_Risk'], df_test['CVD_Risk']])
    label_encoder.fit(all_labels)
    
    for df in [df_train, df_val, df_test]:
        df['CVD_Risk_Encoded'] = label_encoder.transform(df['CVD_Risk'])
    
    # Convert to DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_pandas(df_train),
        "validation": Dataset.from_pandas(df_val),
        "test": Dataset.from_pandas(df_test)
    })
    
    return dataset, label_encoder

# Preprocess function for tokenization
def preprocess_function(examples, tokenizer):
    result = tokenizer(examples["CVD_Input"], padding="max_length", truncation=True)
    result["labels"] = examples["CVD_Risk_Encoded"]
    return result

# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='weighted')
    }

# Main fine-tuning function
def fine_tune_model(train_path, val_path, test_path, model_name='dmis-lab/biobert-v1.1'):
    # Load and prepare data
    dataset, label_encoder = load_data(train_path, val_path, test_path)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        problem_type="single_label_classification"
    )
    
    # Tokenize datasets
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy='epoch',
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none",
        metric_for_best_model='eval_loss'
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate on test set
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(f"Test results: {test_results}")
    
    return model, tokenizer, label_encoder

# Function to predict risk for new input
def predict_risk(input_text, model, tokenizer, label_encoder):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    return label_encoder.inverse_transform([predicted_class])[0]

# Run fine-tuning
if __name__ == "__main__":
    model, tokenizer, label_encoder = fine_tune_model(
        '/path/to/train_cvd.csv',
        '/path/to/val_cvd.csv',
        '/path/to/test_cvd.csv'
    )
    
    # Example prediction
    example_input = "This individual is a Female with an age between Age 55 to 59. ..."
    predicted_risk = predict_risk(example_input, model, tokenizer, label_encoder)
    print(f"Predicted risk: {predicted_risk}")
