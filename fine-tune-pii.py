from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, AdamW, pipeline
from datasets import load_dataset, DatasetDict
from seqeval.metrics import classification_report
import torch
import numpy as np
import evaluate

def generate_sequence_labels(text, privacy_mask):

    # sort privacy mask by start position
    privacy_mask = sorted(privacy_mask, key=lambda x: x['start'], reverse=True)
    
    # replace sensitive pieces of text with labels
    for item in privacy_mask:
        label = item['label']
        start = item['start']
        end = item['end']
        value = item['value']
        # count the number of words in the value
        word_count = len(value.split())
        
        # replace the sensitive information with the appropriate number of [label] placeholders
        replacement = " ".join([f"{label}" for _ in range(word_count)])
        text = text[:start] + replacement + text[end:]
        
    words = text.split()
    # assign labels to each word
    labels = []
    for word in words:
        match = re.search(r"(\w+)", word)  # match any word character
        if match:
            label = match.group(1)
            if label in label_set:
                labels.append(label)
            else:
                # any other word is labeled as "O"
                labels.append("O")
        else:
            labels.append("O")
    return labels


# Main LLM Training function
def FineTuneLLM():

    # Load the pii dataset
    pii_dataset = load_dataset("ai4privacy/pii-masking-400k")

    example = pii_dataset["train"][0]

    # Test with only the first 10000 elements
    train_subset = pii_dataset['train'].select(range(10000))
    validation_subset = pii_dataset['validation'].select(range(10000))

    # Combine the subsets into a new DatasetDict
    pii_dataset = DatasetDict({
        'train': train_subset,
        'validation': validation_subset
    })

    # Iterate over a small sample of the dataset and grab the unique labels
    unique_labels = set()
    for example in pii_dataset['train'].select(range(100)):
        unique_labels.update(example['mbert_token_classes'])

    # Create label to id and id to label mappings
    unique_labels = sorted(unique_labels) # Ensure consistent ordering
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    
    # Load the tokenizer
    model_name = "distilbert/distilbert-base-multilingual-cased"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use a data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Prepare the dataset for training by tokenizing and align the input
    def tokenize_and_align_labels(examples):
        words = [t.split() for t in examples["source_text"]]
        tokenized_inputs = tokenizer(words, truncation=True, is_split_into_words=True, max_length=512)
        source_labels = [
            generate_sequence_labels(text, mask)
            for text, mask in zip(examples["source_text"], examples["privacy_mask"])
        ]

        labels = []
        valid_idx = []
        for i, label in enumerate(source_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # map tokens to their respective word.
            previous_label = None
            label_ids = [-100]
            try:
                for word_idx in word_ids:
                    if word_idx is None:
                        continue
                    elif label[word_idx] == "O":
                        label_ids.append(label2id["O"])
                        continue
                    elif previous_label == label[word_idx]:
                        label_ids.append(label2id[f"I-{label[word_idx]}"])
                    else:
                        label_ids.append(label2id[f"B-{label[word_idx]}"])
                    previous_label = label[word_idx]
                label_ids = label_ids[:511] + [-100]
                labels.append(label_ids)
                # print(word_ids)
                # print(label_ids)
            except:
                global k
                k += 1
                # print(f"{word_idx = }")
                # print(f"{len(label) = }")
                labels.append([-100] * len(tokenized_inputs["input_ids"][i]))

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

def ComputeMetricsLLM():
    
    # Use the validation set to validate the effective mapping of the different categories
    pii_dataset = load_dataset("ai4privacy/pii-masking-400k")
    
    # Load the fine-tuned model and tokenizer
    folder_to_model = "results/my-pii-model"
    model = AutoModelForTokenClassification.from_pretrained(folder_to_model)
    tokenizer = AutoTokenizer.from_pretrained(folder_to_model)

    unique_labels = list(model.config.id2label.values())
    label2id = model.config.label2id
    true_labels = []
    predicted_labels = []

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
    
    tokenized_dataset = pii_dataset['validation'].map(tokenize_and_align_labels, batched=True)

    for entry in tokenized_dataset:
        # Convert input_ids, attention_mask, and labels into tensors
        input_ids = torch.tensor(entry["input_ids"]).unsqueeze(0)  # Add batch dimension
        attention_mask = torch.tensor(entry["attention_mask"]).unsqueeze(0)  # Add batch dimension
        labels = torch.tensor(entry["labels"]).unsqueeze(0)  # Add batch dimension

        # Run the model to get predictions
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = np.argmax(logits.detach().cpu().numpy(), axis=2)

        # Process predictions and true labels to exclude padding tokens (labeled as -100)
        true_label = [unique_labels[l] for l in labels[0].numpy() if l != -100]  # Get true labels excluding padding
        prediction = [unique_labels[p] for p, l in zip(predictions[0], labels[0].numpy()) if l != -100]  # Get predictions excluding padding

        true_labels.append(true_label)
        predicted_labels.append(prediction)

    # Generate the classification report
    report = classification_report(true_labels, predicted_labels, output_dict=True)

    # Display the report as a dictionary with per-category metrics
    for category, scores in report.items():
        print(f"{category}: Precision={scores['precision']:.2f}, Recall={scores['recall']:.2f}, F1={scores['f1-score']:.2f}")

def main():

    # Fine Tune DeBertaV3 with sensitive datasets
    #FineTuneLLM()

    # Evaluate on all of the categories
    #ComputeMetricsLLM()
    

if __name__ == "__main__":
    main()