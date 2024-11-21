from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import easygui
from pynput import keyboard

model_name = "results/my-pii-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def ClassifyPii(text):

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted labels
    predictions = torch.argmax(outputs.logits, dim=-1)
    predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

    print(predicted_token_class)
    return predicted_token_class

def SendMessageAlert(text):

    def show_security_alert():
        response = easygui.buttonbox("⚠️ Security Alert!\n\n"
                                  "You are about to expose sensitive data. "
                                  "Please ensure that you do not share personal information "
                                  "or sensitive details.\n\n"
                                  "Do you wish to continue?",
                                  title="Security Warning",
                                  choices=["Proceed", "Cancel"])
        
    
        # Handle the response
        if response == "Proceed":
            print("User chose to proceed.")
        else:
            print("User chose to cancel.")

    # Get the prediction classes for the message
    predicted_token_classes = ClassifyPii(text)

    # Check to see if this anything else classified besides PIIs
    if any(category != 'O' for category in predicted_token_classes):
        show_security_alert()
    else:
        print(f"Final text: '{text}', is safe")


def ListenToUserInput():

    user_input = ""
    def on_press(key):
        nonlocal user_input
        try:
            if key == keyboard.Key.enter:
                # Get the final message and see if it's safe
                SendMessageAlert(user_input)

                # Reset our string
                user_input = ""
            elif key == keyboard.Key.space:
                user_input += " "
            elif hasattr(key, 'char') and key.char is not None:
                user_input += key.char
            elif key == keyboard.Key.backspace:
                # Handle backspace and remove the last character
                user_input = user_input[:-1]
            else:
                pass
        except AttributeError:
            pass

    print("Ready to capture input. Start typing and press Enter when done.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def main():

    # Create the main entry way in
    ListenToUserInput()


if __name__ == "__main__":
    main()