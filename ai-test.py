import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from huggingface_hub import login
from datasets import load_dataset
import numpy as np
import evaluate

def CheckLlama3(device):
    # Use Llama 3.2 for text generation
    generator = pipeline("text-generation", model="meta-llama/Llama-3.2-1B", device=device)
    print(generator("The best pokemon game of all time is",
          max_length=25,
          num_return_sequences=4))

def TestTransformers():
    # Mask Filling
    unmasker = pipeline('fill-mask', model='bert-base-cased')
    print(unmasker("The best pokemon game is [MASK].", top_k=3))

    # Translation
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")
    print(translator("Hi there, my name is Abhi. Nice to meet you!"))

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    sequence1 = "I’ve been waiting for a HuggingFace course my whole life."
    sequence2 = "I hate this so much!"
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence1)))
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence2)))

def TestHandleMultipleSequences():
    # Handling multiple sequences
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    sequence = "I've been waiting for a HuggingFace course my whole life."

    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    input_ids = torch.tensor([ids])
    output = model(input_ids)
    print(f"Logits: {output.logits}")

    batched_ids = [ids, ids]
    batched_input = torch.tensor(batched_ids)
    outputBatch = model(batched_input)
    print(f"batched logits: {outputBatch.logits}")

def TestModelPadding():
    # Adding Padding
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    sequence1 = "I’ve been waiting for a HuggingFace course my whole life."
    sequence2 = "I hate this so much!"

    # Get the individual logits
    sequence1_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence1))
    sequence2_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence2))
    print(f"seq1 ids: {sequence1_ids}, seq2 ids: {sequence2_ids}")
    print(model(torch.tensor([sequence1_ids])).logits)
    print(model(torch.tensor([sequence2_ids])).logits)

    # Manually pad the sequence
    lengthToPad = (len(sequence1_ids) - len(sequence2_ids))
    sequence2_ids += [tokenizer.pad_token_id] * lengthToPad
    batched_ids = [sequence1_ids, sequence2_ids]
    print(f"batched Ids: {batched_ids}")
    attention_mask = [[1] * len(sequence1_ids), [1 if x > 0 else x for x in sequence2_ids]]
    print(f"attn mask: {attention_mask}")

    # Get the combined logits (should match the individual logits)
    batched_output = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
    print(batched_output.logits)


def TestTokenizer():
    # Processing datasets
    raw_datasets = load_dataset("glue", "mrpc")
    raw_train_dataset = raw_datasets["train"]

    # 15th and 87th element
    print(f"15th: {raw_train_dataset[14]}, 87th: {raw_train_dataset[86]}")

    # Tokenize the two sentences in the 15 training data
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(raw_train_dataset[14]["sentence1"], raw_train_dataset[14]["sentence2"])
    print(inputs)
    print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

def TestTokenizerAndPadding():
    # Processing the data with a different dataset
    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True)

    # Tokenize the dataset
    raw_dataset = load_dataset("Tohrumi/glue_sst2_10k")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

    # Process through a sample of the dataset
    samples = tokenized_dataset["train"][:8]
    samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence"]}
    batch = data_collator(samples)
    test_batch = {k: v.shape for k, v in batch.items()}

def TestFineTuneBERT():
    # Fine tune BERT with GLUE SST
    checkpoint = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    training_args = TrainingArguments("test-trainer")

    # Processing the data with a different dataset
    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True)

    # Tokenize the dataset
    raw_dataset = load_dataset("Tohrumi/glue_sst2_10k")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

    # Create an evaluation function
    def compute_metrics(eval_preds):
        metric = evaluate.load("Tohrumi/glue_sst2_10k")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    # Run the training sequence with the evaluation
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['eval'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

# Define a main function that serves as the entry point of the program
def main():

    # See if GPU is available to use 
    device = 0 if torch.cuda.is_available() else -1
    print(torch.cuda.is_available())

    # Test the different Hugging Face functions here


# Ensure that the main function is called when the script is executed directly
if __name__ == "__main__":
    main()