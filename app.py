from pathlib import Path
import runpy

APP_ENTRY = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
runpy.run_path(str(APP_ENTRY), run_name="__main__")
