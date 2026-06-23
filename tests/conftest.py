import sys
from pathlib import Path

# Make the src layout importable without an editable install (keeps CI simple).
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
