import win32gui
import win32process
import psutil
from pywinauto import Application, Desktop
from pynput import keyboard
from keyboard import is_pressed
import time
import pyperclip

from find_window import FindWindow

def get_foreground_window_app():
    """Get the name of the app in the foreground window."""
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

def send_redacted_input():
    # Find the window that ChatGPT is open in 
    app = Application(backend='uia').connect(title="ChatGPT")
    main_window = app.window(title="ChatGPT")
    if main_window.exists():
        # Grab the text that's in the window by simulating a copy and paste
        main_window.type_keys("^a^c")  
        copied_text = pyperclip.paste()
        print(f"Copied text: {copied_text}")

        # Redact the text from our LLM 
        main_window.type_keys("^a{BACKSPACE}REDACTED")

def handle_chatgpt_input():

    def on_press(key):
        try:
            if key == keyboard.Key.down:
                # Grab the message and send the redacted version
                send_redacted_input()
            else:
                pass
        except AttributeError:
            pass

    print("Ready to capture input. Start typing and press Enter when done.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def main():
    print("Tracking active application... Press Ctrl+C to stop.")
    try:
        while True:
            # Grab the current window and check if it's ChatGPT
            active_app = get_foreground_window_app()
            if active_app and "ChatGPT" in active_app:
                print("ChatGPT is active! Taking action...")
                handle_chatgpt_input()

            # Check every second
            time.sleep(1)  
    except KeyboardInterrupt:
        print("\nStopped tracking.")

if __name__ == "__main__":
    main()