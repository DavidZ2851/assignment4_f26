"""Screenshot a GT/prediction geometry viewer (index.html) with headless Chrome, choosing the scene and geometry.
  python tools/screenshot_viewer.py <index.html> <out.png> [--scene 0] [--geometry gt|prediction] [--size 1700,1300]
A temporary copy of the page selects the scene/geometry on load (the original page is not modified).
"""

import argparse
import pathlib
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("out")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--geometry", default="gt")
    ap.add_argument("--size", default="1700,1300")
    a = ap.parse_args()
    page = pathlib.Path(a.html).resolve()
    hook = (
        "<script>window.addEventListener('load',()=>setTimeout(()=>{"
        f"const s=document.getElementById('scene');if(s){{s.value='{a.scene}';s.onchange&&s.onchange();}}"
        f"const g=document.getElementById('geometry');if(g&&!g.hidden){{g.value='{a.geometry}';g.onchange&&g.onchange();}}"
        "},300));</script>"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=page.parent, delete=False) as f:
        f.write(page.read_text().replace("</body>", hook + "</body>"))
        tmp = pathlib.Path(f.name)
    try:
        subprocess.run(
            ["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
             f"--window-size={a.size}", "--virtual-time-budget=20000",
             f"--screenshot={pathlib.Path(a.out).resolve()}", tmp.as_uri()],
            check=True, capture_output=True, timeout=180,
        )
    finally:
        tmp.unlink()


if __name__ == "__main__":
    main()
