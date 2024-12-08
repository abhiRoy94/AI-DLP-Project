import win32gui
import win32process
import psutil
from pywinauto import Application, Desktop
from pynput import keyboard
import keyboard as kb
import time
import pyperclip
import easygui

# Using a separate model until mine is fully trained
from piiranha_model import PiiranhaModel

# Track if 'enter' key is blocked
blocked_keys = set()

def validate_input_text(text):
    llm_model = PiiranhaModel()
    return llm_model.mask_pii(text, aggregate_redaction=True)

def show_security_alert(main_window, masked_text):
    response = easygui.buttonbox("⚠️ Security Alert!\n\n"
                                "You are about to expose sensitive data. "
                                "Please ensure that you do not share personal information "
                                "or sensitive details.\n\n"
                                "Do you wish to continue?",
                                title="Security Warning",
                                choices=["Proceed", "Send Redacted Version"])
    
    # Handle the response
    if response == "Proceed":
        print("User chose to proceed.")
    else:
        print("User chose to redact message")
        main_window.type_keys("^a{BACKSPACE}")
        main_window.type_keys(masked_text, with_spaces=True)

    kb.send('enter')

def validate_input():

    # Find the window that ChatGPT is open in 
    app = Application(backend='uia').connect(title="ChatGPT")
    main_window = app.window(title="ChatGPT")

    copied_text = ""
    if main_window.exists():
        # Grab the text that's in the window by simulating a copy and paste
        main_window.type_keys("^a^c")  
        copied_text = pyperclip.paste()
        print(f"Copied text: {copied_text}")
    else:
        print("Could not find ChatGPT window.")
        return

    # Check model output and see if it's sensitive based on the LLM output
    new_text = validate_input_text(copied_text)
    if new_text != copied_text:
        show_security_alert(main_window, new_text)
    else:
        kb.send('enter')

def get_foreground_window_app():
    # Get the handle of the foreground window
    hwnd = win32gui.GetForegroundWindow()
    
    # Get the process ID of the app owning the foreground window
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    
    # Get the process name from the PID
    try:
        process = psutil.Process(pid)
        app_name = process.name()  # e.g., "ChatGPT.exe"
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        app_name = None

    return app_name

def on_press(key):
    try:
        if key == keyboard.Key.enter:
            # Validate the input and send redacted or original version based on the user's choice
            unblock_key_with_tracking('enter')
            validate_input()
            return False
        else:
            pass
    except AttributeError:
        pass

def block_key_with_tracking(key):
    # Block the key and add it to the blocked_keys set
    kb.block_key(key)
    blocked_keys.add(key)

def unblock_key_with_tracking(key):
    if key in blocked_keys:
        # Unblock the key and remove it from the blocked_keys set
        kb.unblock_key(key)
        blocked_keys.remove(key)

def main():
    print("Tracking active application... Press Ctrl+C to stop.")
    listener = None
    try:
        while True:
            # Grab the current window and check if it's ChatGPT
            active_app = get_foreground_window_app()
            if active_app and "ChatGPT" in active_app:
                # Block the 'enter' key if we're dealing with a ChatGPT window
                if 'enter' not in blocked_keys:  
                    block_key_with_tracking('enter')
                if listener is None or not listener.running:
                    listener = keyboard.Listener(on_press=on_press)
                    listener.start()  
            else:
                # Keep the 'enter' key useable if we're not on a chatGPT window
                if 'enter' in blocked_keys: 
                    unblock_key_with_tracking('enter')
                if listener is not None and listener.running:
                    listener.stop()  

            # Check every second
            time.sleep(1)  
    except KeyboardInterrupt:
        print("\nStopped tracking.")

if __name__ == "__main__":
    main()