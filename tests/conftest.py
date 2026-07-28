"""Make the repo root importable so tests can import the modules under test.

This replaces the per-file `sys.path.insert` each test used to carry. It stays
until the package layout lands, at which point the imports become absolute and
this file can go.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
