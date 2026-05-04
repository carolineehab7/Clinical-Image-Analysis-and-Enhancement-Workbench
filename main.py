"""
Clinical Image Analysis & Enhancement Workbench
Phase 1 — Spatial Domain Operations & Core Architecture

Run:  python main.py
"""

import sys
import os

# Ensure the project root is on the path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import MainWindow


def main():
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
