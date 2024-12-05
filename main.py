import win32gui
import win32process
import psutil
from pywinauto import Application, Desktop
from pynput import keyboard
from keyboard import is_pressed
import time

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

def ListenToUserInput():
    user_input = ""
    def on_press(key):
        nonlocal user_input
        try:
            if key == keyboard.Key.enter:
                # Get the final message and see if it's safe
                print(f"user input: {user_input}")
                user_input = ""
                return False
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

def handle_chatgpt_input():

    # Find the text input window in ChatGPT
    fw = FindWindow("ChatGPT")
    fw.find_window_by_name()
    #ListenToUserInput()

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