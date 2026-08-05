"""Put the project root on sys.path so `import src.probes` works from anywhere."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
