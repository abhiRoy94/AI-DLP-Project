import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "iiiorg/piiranha-v1-detect-personal-information"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class PiiranhaModel():

    def mask_pii(self, text, aggregate_redaction=True):
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get the model predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted labels
        predictions = torch.argmax(outputs.logits, dim=-1)

        # Convert token predictions to word predictions
        encoded_inputs = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True)
        offset_mapping = encoded_inputs['offset_mapping']

        masked_text = list(text)
        is_redacting = False
        redaction_start = 0
        current_pii_type = ''

        for i, (start, end) in enumerate(offset_mapping):
            if start == end:  # Special token
                continue

            label = predictions[0][i].item()
            if label != model.config.label2id['O']:  # Non-O label
                pii_type = model.config.id2label[label]
                if not is_redacting:
                    is_redacting = True
                    redaction_start = start
                    current_pii_type = pii_type
                elif not aggregate_redaction and pii_type != current_pii_type:
                    # End current redaction and start a new one
                    self.apply_redaction(masked_text, redaction_start, start, current_pii_type, aggregate_redaction)
                    redaction_start = start
                    current_pii_type = pii_type
            else:
                if is_redacting:
                    self.apply_redaction(masked_text, redaction_start, end, current_pii_type, aggregate_redaction)
                    is_redacting = False

        # Handle case where PII is at the end of the text
        if is_redacting:
            self.apply_redaction(masked_text, redaction_start, len(masked_text), current_pii_type, aggregate_redaction)

        return ''.join(masked_text)

    def apply_redaction(self, masked_text, start, end, pii_type, aggregate_redaction):
        for j in range(start, end):
            masked_text[j] = ''
        if aggregate_redaction:
            masked_text[start] = '[redacted]'
        else:
            masked_text[start] = f'[{pii_type}]'