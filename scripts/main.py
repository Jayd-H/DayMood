import sys
import os

current_dir = os.path.dirname(__file__)
scripts_dir = os.path.join(current_dir)
sys.path.append(scripts_dir)

from gui import create_gui

if __name__ == "__main__":
    create_gui()
