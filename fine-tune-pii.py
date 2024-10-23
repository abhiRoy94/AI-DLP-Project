from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Main LLM Training function
def FineTuneLLM():

    # Load the pii dataset
    pii_dataset = load_dataset("ai4privacy/pii-masking-400k")
    print(pii_dataset.column_names["train"])

    # Iterate over a small sample of the dataset and grab the unique labels
    unique_labels = set()
    for example in pii_dataset['train'].select(range(100)):
        unique_labels.update(example['mbert_token_classes'])
    #print(f"labels: {unique_labels}, length: {len(unique_labels)}")

    # Create label to id and id to label mappings
    unique_labels = sorted(unique_labels) # Ensure consistent ordering
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    # Load the tokenizer
    model_name = "microsoft/deberta-v3-base"  
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    # Use a data collator
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)
    
    # Prepare the dataset for training
    def tokenize_and_label(examples):
        # Tokenize the masked text
        encodings = tokenizer(
            examples['masked_text'],
            truncation=True,
            padding='max_length',  # or 'longest' based on your preference
            max_length=128,  # Set to your desired max length
            return_offsets_mapping=True,  # Useful for aligning labels
        )

        # Create labels from mbert_token_classes
        labels = []
        for i, label_sequence in enumerate(examples['mbert_token_classes']):
            # Assuming label_sequence is in string format, you might need to convert them to IDs
            # For example, if your labels are in string format, create a label mapping
            # Example: label2id = {"LABEL_A": 0, "LABEL_B": 1, ...}
            label = label2id[label_sequence]  # Adjust if necessary
            labels.append(label)

        # Assign the labels to the encoding
        encodings['labels'] = labels
        return encodings
    
    tokenized_dataset = pii_dataset.map(tokenize_and_label, batched=True)

    print(tokenized_dataset['train'][0])

    # Load the model and ensure it runs on the GPU
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        label2id=label2id,
        id2label=id2label
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    '''
    training_args = TrainingArguments(
        output_dir='./results/my-pii-model',         # Output directory for model predictions and checkpoints
        evaluation_strategy="epoch",                 # Evaluate at the end of each epoch
        learning_rate=5e-5,                          # Learning rate
        per_device_train_batch_size=128,             # Training batch size
        per_device_eval_batch_size=128,              # Evaluation batch size
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    '''



# Compute metrics functions that will be used during training
def compute_metrics(pred):
    # Extract predictions and true labels
    predictions, labels = pred

    # Get the predicted classes
    preds = np.argmax(predictions, axis=1)

    # Mask the padding tokens (usually labeled as -100)
    mask = labels != -100  # Assuming -100 is used for padding
    labels = labels[mask]
    preds = preds[mask]

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

# Define a main function that serves as the entry point of the program
def main():

    # Fine Tune Llama 3 with sensitive datasets
    FineTuneLLM()


# Ensure that the main function is called when the script is executed directly
if __name__ == "__main__":
    main()