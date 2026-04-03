"""Export brats2020unet2D.ipynb to HTML.

Usage:
    python scripts/export_html.py                  # default output next to notebook
    python scripts/export_html.py -o output.html   # custom output path
"""
import subprocess
import sys
import os
import json
import tempfile
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(REPO_ROOT, "brats2020unet2D.ipynb")


def strip_widget_metadata(nb_path, out_path):
    """Copy notebook, removing broken widget metadata that crashes nbconvert."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            md = output.get("metadata", {})
            if "widgets" in md:
                del md["widgets"]
            data = output.get("data", {})
            data.pop("application/vnd.jupyter.widget-view+json", None)

    # Also strip notebook-level widget state
    nb_meta = nb.get("metadata", {})
    nb_meta.pop("widgets", None)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export notebook to HTML")
    parser.add_argument("-o", "--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    output = args.output or os.path.join(REPO_ROOT, "brats2020unet2D.html")

    # Create a clean temp copy without widget metadata
    tmp = tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False)
    tmp.close()
    try:
        strip_widget_metadata(NOTEBOOK, tmp.name)
        cmd = [
            sys.executable, "-m", "nbconvert",
            "--to", "html",
            "--output", os.path.abspath(output),
            tmp.name,
        ]
        print(f"Exporting {NOTEBOOK} -> {output}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            sys.exit(result.returncode)
        print(f"Done: {output}")
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    main()