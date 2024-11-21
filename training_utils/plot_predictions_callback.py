import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from transformers import TrainerCallback, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

class PlotPredictionsCallback(TrainerCallback):
    def __init__(self, tokenizer, validation_dataset):
        self.tokenizer = tokenizer
        self.validation_dataset = validation_dataset

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """ This method is called during evaluation after every evaluation phase. """
        # Get model predictions on validation set
        predictions, label_ids, metrics = kwargs['trainer'].predict(self.validation_dataset)
        
        # Convert predictions and labels into tokens and labels
        predictions = torch.argmax(torch.tensor(predictions), dim=-1)

        decoded_predictions = []
        decoded_labels = []

        for i, (input_ids, label_ids) in enumerate(zip(predictions, label_ids)):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            label_tokens = self.tokenizer.convert_ids_to_tokens(label_ids)

            # Filter out special tokens like [CLS], [SEP], etc.
            tokens = [token for token in tokens if token not in self.tokenizer.all_special_tokens]
            label_tokens = [label for label in label_tokens if label not in self.tokenizer.all_special_tokens]

            decoded_predictions.append(tokens)
            decoded_labels.append(label_tokens)

        # Plot predictions for the first few samples
        self.plot_predictions(decoded_predictions[:3], decoded_labels[:3])

        # Calculate and print classification report
        report = classification_report(np.concatenate(label_ids), np.concatenate(predictions), output_dict=True)
        print("Classification report:", report)

    def plot_predictions(self, predictions, labels):
        """ Helper method to plot predictions against labels """
        for i, (text, pred, true) in enumerate(zip(predictions, labels)):
            plt.figure(figsize=(10, 5))
            plt.title(f"Sample {i + 1}")
            for t, p, l in zip(text, pred, true):
                color = 'green' if p == l else 'red'
                plt.text(0.5, 0.5, f'{t}: {l} -> {p}', color=color)
            plt.show()