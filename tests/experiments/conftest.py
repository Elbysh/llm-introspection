import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "utils"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
