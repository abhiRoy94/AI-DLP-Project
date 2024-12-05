import win32gui
import win32con
import win32api
from pywinauto import Application, Desktop
from pywinauto.findwindows import find_elements

class FindWindow():

    def __init__(self, window_name):
        elements = find_elements(title="ChatGPT", backend="win32", visible_only=False)
        for e in elements:
            print(f"Title: {e.name}, Process ID: {e.process_id}, Handle: {e.handle}")
        self.window1 = Application(backend="win32").connect(handle=26084676)
        self.window2 = Application(backend="win32").connect(handle=1836486)
        self.main_window = self.window2.window(title=window_name)
        #self.app = Application(backend="win32").connect(title=window_name)
        #self.app = Desktop(backend="win32").window(title=window_name)
        #self.main_window = self.app.window(title=window_name)  # Make sure the window title is correct

    def find_window_by_name(self):
        #print(self.main_window.print_control_identifiers())
        #print(self.app['Pane0'].print_control_identifiers())
        #print(self.app["Intermediate D3D Window"].)
        #print(self.app)
        #print(self.app.windows)
        print(self.window1.windows.window_text())