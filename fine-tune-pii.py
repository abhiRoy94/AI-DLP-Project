from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, AdamW, pipeline
from datasets import load_dataset
import torch
import numpy as np
import evaluate

# Main LLM Training function
def FineTuneLLM():

    # Load the pii dataset
    pii_dataset = load_dataset("ai4privacy/pii-masking-400k")

    # Iterate over a small sample of the dataset and grab the unique labels
    unique_labels = set()
    for example in pii_dataset['train'].select(range(100)):
        unique_labels.update(example['mbert_token_classes'])

    # Create label to id and id to label mappings
    unique_labels = sorted(unique_labels) # Ensure consistent ordering
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    
    # Load the tokenizer
    model_name = "microsoft/deberta-v3-small"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use a data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Prepare the dataset for training by tokenizing and align the input
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["source_text"], truncation=True)
        currIds = []
        for i in range(len(examples["mbert_token_classes"])):
            currIds.append([label2id[example] for example in examples["mbert_token_classes"][i]])
        labels = []
        for i, label in enumerate(currIds):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    tokenized_pii = pii_dataset.map(tokenize_and_align_labels, batched=True)

    # Compute metrics functions that will be used during training
    seqeval = evaluate.load("seqeval")
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [unique_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [unique_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    torch.cuda.empty_cache()
    # Load the model and ensure it runs on the GPU
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        label2id=label2id,
        id2label=id2label
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    
    training_args = TrainingArguments(
        output_dir='./results/my-pii-model',         # Output directory for model predictions and checkpoints
        save_strategy="epoch",
        evaluation_strategy="epoch",                 # Evaluate at the end of each epoch
        learning_rate=5e-5,                          # Learning rate
        per_device_train_batch_size=16,              # Training batch size
        per_device_eval_batch_size=16,               # Evaluation batch size
        num_train_epochs=5,                          # Number of training epochs
        weight_decay=0.01,                           # Weight decay for optimizer
        logging_dir='./logs',                        # Directory for storing logs
        logging_steps=10,                            # Log every 10 steps
        seed=42,                                     # Seed for reproducibility
        fp16=True,                                   # Enable mixed precision training (Native AMP)
        optim="adamw_torch",                         # Use AdamW optimizer with PyTorch defaults
        adam_beta1=0.9,                              # First beta for Adam optimizer
        adam_beta2=0.999,                            # Second beta for Adam optimizer
        adam_epsilon=1e-8,                           # Epsilon for Adam optimizer
        lr_scheduler_type="linear",                  # Learning rate scheduler type
        warmup_ratio=0.05,                           # Warmup ratio for learning rate scheduler
    )

    optimizer = AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_pii['train'],
        eval_dataset=tokenized_pii['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )

    trainer.train()

def main():

    # Fine Tune DeBertaV3 with sensitive datasets
    FineTuneLLM()

if __name__ == "__main__":
    main()