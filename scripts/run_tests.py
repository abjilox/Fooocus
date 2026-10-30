"""Run unit tests without passing unittest flags to Fooocus's application parser."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = ['fooocus-tests']
suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
