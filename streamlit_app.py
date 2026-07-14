"""Repository root entry point for the Streamlit dashboard.

Streamlit Community Cloud runs the file at the repository root, but the app lives at
src/ercot_bess/dashboard/app.py and imports the ercot_bess package absolutely. This shim puts
src on the import path and then runs the app script fresh on every rerun, run_path with a main
run name, which is exactly how Streamlit runs a top level script. So the dashboard behaves the
same locally and when deployed, with no editable install needed on the host.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

runpy.run_path(str(_SRC / "ercot_bess" / "dashboard" / "app.py"), run_name="__main__")
