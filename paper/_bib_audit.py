import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
with open(ROOT / "neurips_submission.tex", encoding="utf-8") as f:
    tex = f.read()
with open(ROOT / "neurips_refs.bib", encoding="utf-8") as f:
    bib = f.read()

cited = set()
for m in re.finditer(r"\\cite[tp]?\*?\{([^}]+)\}", tex):
    for k in m.group(1).split(","):
        cited.add(k.strip())

defined = set()
for m in re.finditer(r"(?m)^@\w+\{([^,\s]+),", bib):
    defined.add(m.group(1))

missing = cited - defined
unused = defined - cited
print(f"Cited keys: {len(cited)}")
print(f"Defined keys: {len(defined)}")
print(f"\nMISSING (cited but not in bib): {len(missing)}")
for k in sorted(missing):
    print(f"  - {k}")
print(f"\nUNUSED (in bib but not cited): {len(unused)}")
for k in sorted(unused):
    print(f"  - {k}")

# Structural checks
entries = re.split(r"(?m)^@", bib)
problems = []
for e in entries:
    if not e.strip():
        continue
    e = "@" + e
    m = re.match(r"@(\w+)\{([^,\s]+),", e)
    if not m:
        problems.append(("PARSE", e[:80]))
        continue
    kind, key = m.group(1).lower(), m.group(2)
    body = e[m.end():]
    bl = body.lower()
    if "year" not in bl:
        problems.append(("NO YEAR", key))
    if "title" not in bl:
        problems.append(("NO TITLE", key))
    if "author" not in bl:
        problems.append(("NO AUTHOR", key))
    if kind == "inproceedings" and "booktitle" not in bl:
        problems.append(("@inproceedings missing booktitle", key))
    if kind == "article" and "journal" not in bl:
        problems.append(("@article missing journal", key))
    if re.search(r"note=\{[^}]*INCOMPLETE", body, re.I):
        problems.append(("INCOMPLETE note remains", key))
    if re.search(r"author=\{[^}]*\bothers\b", body):
        problems.append(('author uses "others"', key))
    if re.search(r"journal=\{arXiv preprint\}", body):
        problems.append(('journal="arXiv preprint" no ID', key))
    if re.search(r"journal=\{arXiv preprint\},", body) and "arXiv:" not in body:
        problems.append(("arXiv preprint without ID", key))

print(f"\nSTRUCTURAL ISSUES: {len(problems)}")
for kind, k in problems:
    print(f"  [{kind}] {k}")
