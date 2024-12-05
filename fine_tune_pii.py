from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, AdamW
from datasets import load_dataset, DatasetDict
import numpy as np
import torch
import evaluate
from seqeval.metrics import classification_report
from seqeval.metrics import precision_score, recall_score, f1_score
import sys
import os

from training_utils.plot_losses_callback import PlotLossesCallback
from training_utils.plot_predictions_callback import PlotPredictionsCallback

# Define global variables used in separate functions
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-multilingual-cased")
special_token_ids = tokenizer.all_special_ids
label2id, id2label = {}, {}
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds, id2label):
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

def create_small_subset(dataset, train_size=5000, val_size=1000):
    small_train = dataset["train"].select(range(train_size))
    small_val = dataset["validation"].select(range(val_size))

    dataset_dict = DatasetDict({
        "train": small_train,
        "validation": small_val
    })
        
    return dataset_dict

def split_dataset(dataset):
    # Split the original dataset into 80% train and 20% test
    split_data = dataset["train"].train_test_split(train_size=0.8, seed=42)
    
    # Split the 20% test data into 50% validation and 50% test (i.e., 10% of the original data for each)
    validation_split = split_data["test"].train_test_split(train_size=0.5, seed=42)

    # If the dataset already contains a validation split, use it as is
    dataset_dict = DatasetDict({
        "train": split_data["train"],         # 80% of the original data for training
        "validation": dataset["validation"],  # Keep the existing validation split
        "test": validation_split["test"]      # 10% of the original data for testing
    })
        
    return dataset_dict

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
    
    # 1. Gather the dataset
    dataset = load_dataset("ai4privacy/pii-masking-400k")
    #new_pii_dataset = split_dataset(dataset) # Use this if we're planning on splitting the dataset into a train, validation, test split
    new_pii_dataset = dataset

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

    '''
    training_args = TrainingArguments(
        output_dir='./results/my_pii_model',            # Directory to save the model checkpoints
        evaluation_strategy="epoch",                    # Evaluate every 'eval_steps' epochs
        save_strategy="epoch",                          # Save checkpoint every 'save_steps'
        logging_dir='./logs',                           # Directory for log files
        logging_steps=500,                              # Log training progress every 500 steps
        save_steps=1000,                                # Save a checkpoint every 1000 steps
        eval_steps=1000,                                # Run evaluation every 1000 steps
        per_device_train_batch_size=8,                  # Batch size for training on each device
        per_device_eval_batch_size=8,                   # Batch size for evaluation
        gradient_accumulation_steps=2,                  # Accumulate gradients over multiple steps before updating weights
        num_train_epochs=5,                             # Number of training epochs
        weight_decay=0.01,                              # Regularization (L2 weight decay)
        learning_rate=2e-5,                             # Learning rate for the optimizer
        warmup_steps=500,                               # Number of warmup steps for learning rate scheduler
        logging_first_step=True,                        # Log the first training step
        load_best_model_at_end=True,                    # Load the best model based on evaluation metric at the end of training
        metric_for_best_model="eval_loss",              # Use validation loss to identify the best model
        disable_tqdm=False,                             # Enable progress bars
        fp16=True,                                      # Use mixed precision for faster training (requires GPU with FP16 support)
        dataloader_num_workers=4,                       # Number of subprocesses to use for data loading
        run_name="fine_tuning_pii",                     # Name of the run for tracking purposes
        seed=42,                                        # Random seed for reproducibility
        push_to_hub=True,                               # Push model to Hugging Face Hub
        hub_model_id="AyyRoy/my-pii-model",             # Specify your model repo name
        hub_strategy="every_save",                      # Push to the hub every time the model is saved
    )
    '''

    training_args = TrainingArguments(
        output_dir="./results/my_pii_model",  # Output directory for model checkpoints and logs
        evaluation_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        logging_dir="./logs",  # Directory for logs
        logging_steps=500,  # Log every 500 steps
        per_device_train_batch_size=1,  # Batch size per device during training
        per_device_eval_batch_size=1,  # Batch size for evaluation
        gradient_accumulation_steps=4,  # Simulate a larger batch size by accumulating gradients
        num_train_epochs=3,  # Number of epochs
        learning_rate=3e-5,  # Initial learning rate
        warmup_steps=500,  # Warmup steps for learning rate scheduler
        weight_decay=0.01,  # Weight decay
        fp16=True,  # Enable mixed precision training for faster computations
        save_total_limit=2,  # Limit the number of checkpoints saved
        push_to_hub=True,  # Automatically upload the model to the Hugging Face Hub
        hub_model_id="AyyRoy/my-pii-model",  # Replace with your model's name on the Hugging Face Hub
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_loss",  # Metric to select the best model
        greater_is_better=False,  # Minimize the evaluation metric
    )

    model.gradient_checkpointing_enable()

    # Optional: Setup callbacks to measure metrics during training phase
    #plot_losses_callback = PlotLossesCallback()
    #plot_predictions_callback = PlotPredictionsCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator, 
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, id2label)
    )

    # 6. Run training script
    trainer.train()


def main():

    # Call the main fine tuning function
    FineTunePii()


if __name__ == "__main__":
    main()