from transformers import AutoTokenizer
from datasets import load_dataset
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

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

        total_labels.append(token_labels)

    '''
    print(f"offsets: {offsets}")
    print(f"mask: {privacy_mask}")
    print(f"tokens: {tokens}")
    print(f"tokens to mapping: {tokenizer.convert_ids_to_tokens(tokens)}")
    print(f"token labels: {token_labels}")
    '''

    tokenized_output["labels"] = total_labels
    return tokenized_output

# Main fine-tuning function
def FineTunePii():
    
    # 1. Gather the dataset and split it into 'train', 'validation' and 'test'
    dataset = load_dataset("ai4privacy/pii-masking-400k")
    pii_dataset = dataset["train"].train_test_split(train_size=0.8, seed=42)
    pii_dataset["validation"] = dataset['validation']
    pii_dataset["test"] = pii_dataset.pop("test")
    print(pii_dataset)


    # 2. Gather a tokenized and alligned dataset that contains the tokens given to the model, and the associated labels
    tokenized_datasets = pii_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=pii_dataset["train"].column_names)

def main():

    # Call the main fine tuning function
    FineTunePii()


if __name__ == "__main__":
    main()