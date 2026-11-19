# conftest.py (project root)
# Adds the project root to sys.path so pytest can import backend, ml, etc.
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
