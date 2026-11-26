"""Refresh gallery.html; if tools/overnight.flag exists, afterwards run the
overnight pipeline (tools/overnight.py) once and remove the flag."""
import os, runpy, subprocess, sys
from pathlib import Path
here = Path(__file__).resolve().parent
sys.argv = [str(here / "make_gallery_core.py")] + sys.argv[1:]
runpy.run_path(str(here / "make_gallery_core.py"), run_name="__main__")
flag = here / "overnight.flag"
if flag.exists():
    flag.unlink()
    subprocess.run([sys.executable, "-u", str(here / "overnight.py")], cwd=str(here.parent))
