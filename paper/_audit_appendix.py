import re
from pathlib import Path

with open(Path(__file__).parent / "neurips_submission.tex", encoding="utf-8") as f:
    tex = f.read()

app_start = tex.find(r"\appendix")
if app_start < 0:
    raise SystemExit("no appendix marker")

app = tex[app_start:]
end = app.find(r"\end{document}")
if end >= 0:
    app = app[:end]

parts = re.split(r"(\\(?:section|subsection)\{[^}]+\})", app)
counts = []
cur = None
buf = ""
for p in parts:
    m = re.match(r"\\(?:section|subsection)\{([^}]+)\}", p)
    if m:
        if cur is not None:
            counts.append((cur, len(buf.strip())))
        cur = m.group(1)
        buf = ""
    else:
        buf += p
if cur is not None:
    counts.append((cur, len(buf.strip())))

print(f"{'chars':>7}  section")
print("-" * 80)
for name, n in counts:
    flag = "  <<< THIN" if n < 500 else ("  <<< empty-ish" if n < 100 else "")
    print(f"{n:>7}  {name[:68]}{flag}")
