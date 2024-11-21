from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, AdamW
from datasets import load_dataset
import numpy as np
import torch
import evaluate
from seqeval.metrics import classification_report
from seqeval.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

#from plot_losses_callback import PlotLossesCallback

# Define global variables used in separate functions
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-multilingual-cased")
special_token_ids = tokenizer.all_special_ids
label2id, id2label = {}, {}
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Calculate overall metrics
    overall_precision = precision_score(true_labels, true_predictions)
    overall_recall = recall_score(true_labels, true_predictions)
    overall_f1 = f1_score(true_labels, true_predictions)

    # Per-category metrics using classification_report
    report = classification_report(true_labels, true_predictions, output_dict=True)

    # Extract per-category metrics
    category_metrics = {
        category: {
            "precision": report[category]["precision"],
            "recall": report[category]["recall"],
            "f1": report[category]["f1-score"]
        }
        for category in report if category not in {"accuracy", "macro avg", "weighted avg"}
    }

    # Combine all metrics
    metrics = {
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "per_category": category_metrics
    }

    return metrics

def split_dataset(dataset):
    new_dataset = dataset["train"].train_test_split(train_size=0.8, seed=42)
    new_dataset["validation"] = dataset['validation']
    new_dataset["test"] = new_dataset.pop("test")
    return new_dataset

def align_labels_with_tokens(token_labels):
    id_labels = []
    for token in token_labels:
        if token == None:
            id_labels.append(-100)
        else:
            id_labels.append(label2id[token])

    return id_labels

def tokenize_and_align_labels(example):
    
    # Tokenize the input and keep track of the original offset indices
    tokenized_output = tokenizer(example['source_text'], return_offsets_mapping=True, truncation=True)
    tokens = tokenized_output['input_ids']
    offsets = tokenized_output["offset_mapping"]

    # Run through the privacy mask and map each mask to the new tokenized data. Sort the privacy mask to save time. O(nlog(n)) < O(n^2)
    privacy_masks = example['privacy_mask']
    
    total_labels = []
    for i, privacy_mask in enumerate(privacy_masks):
        privacy_mask = sorted(privacy_mask, key=lambda x: x['start'])
        priv_ind = 0
        token_labels = ["O"] * len(tokens[i])
        for j, (start_ind, end_ind) in enumerate(offsets[i]):
            # Check if we're dealing with a special token
            if tokens[i][j] in special_token_ids:
                token_labels[j] = None
                continue

            # We've gone beyond our privacy mask array, so everything else will not be sensitive data
            if (priv_ind) >= len(privacy_mask):
                continue
                
            token_start, token_end = privacy_mask[priv_ind]['start'], privacy_mask[priv_ind]['end']
            token_label = privacy_mask[priv_ind]['label']

            # The current token is part of what our privacy mask detects is sensitive
            if (start_ind >= token_start and end_ind <= token_end):
                token_labels[j] = f"B-{token_label}" if token_start == start_ind else f"I-{token_label}"

            # Move our privacy index up if our end index is greater than the privacy start index of where we are in our list
            if end_ind > token_end:
                priv_ind += 1

        # Convert the labels to ids
        id_labels = align_labels_with_tokens(token_labels)
        total_labels.append(id_labels)

    del tokenized_output['offset_mapping']
    tokenized_output["labels"] = total_labels
    return tokenized_output

def create_label_id_mapping(sample):
    # Grab the unique labels with a set from the given token classes column
    unique_labels = set()
    for tokens in sample['mbert_token_classes']:
        for token in tokens:
            unique_labels.add(token)

    # Split the unique labels into label2id and id2label
    unique_labels = sorted(unique_labels)
    for idx, label in enumerate(unique_labels):
        label2id[label] = idx
        id2label[idx] = label

def FineTunePii():
    
    # 1. Gather the dataset and split it into 'train', 'validation' and 'test'
    dataset = load_dataset("ai4privacy/pii-masking-400k")
    new_pii_dataset = split_dataset(dataset)

    # 2. Create id2Label and label2Id Mapping with a small subset of the dataset
    create_label_id_mapping(new_pii_dataset["train"].select(range(1000)))

    # 3. Gather a tokenized and alligned dataset that contains the tokens given to the model, and the associated labels
    tokenized_datasets = new_pii_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=new_pii_dataset["train"].column_names)

    # 4. Setup a data collator to make batching and padding simple
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 5. Set up the training arguments and environment
    model = AutoModelForTokenClassification.from_pretrained(
        "distilbert/distilbert-base-multilingual-cased",
        id2label=id2label,
        label2id=label2id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./test_training/",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,  # Shortened for testing
        evaluation_strategy="steps",
        eval_steps=100,  # Frequent evaluation
        save_strategy="no",  # Skip saving to speed up testing
        logging_steps=50,
        warmup_steps=10,
        fp16=True,
        seed=42
    )

    # Take a subset of the dataset
    small_train_dataset = tokenized_datasets["train"].select(range(5000))
    small_eval_dataset = tokenized_datasets["validation"].select(range(200))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator, 
        compute_metrics=compute_metrics, 
        #callbacks=[PlotLossesCallback()]
    )

    # 6. Run training script
    trainer.train()


def main():

    # Call the main fine tuning function
    FineTunePii()


if __name__ == "__main__":
    main()