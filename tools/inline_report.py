"""Embed every img/*.png referenced by the report into a single self-contained HTML file."""

import base64
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])  # report/index.html (references img/...)
dst = Path(sys.argv[2])
html = src.read_text()


def embed(m):
    data = base64.b64encode((src.parent / m.group(1)).read_bytes()).decode()
    return f'src="data:image/png;base64,{data}"'


out = re.sub(r'src="(img/[^"]+\.png)"', embed, html)
assert not any(s.startswith("img/") for s in re.findall(r'src="([^"]*)"', out))
dst.write_text(out)
print(f"{dst}: {dst.stat().st_size / 1e6:.1f} MB")
