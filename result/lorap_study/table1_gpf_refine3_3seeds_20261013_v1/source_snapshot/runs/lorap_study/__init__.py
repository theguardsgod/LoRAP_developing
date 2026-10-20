"""Isolated LoRAP experiments; legacy experiment files remain unchanged."""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "lorap-matplotlib"))

