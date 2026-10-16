"""Fail if any placeholder string remains in the notebook text (code and markdown cells).

Run before every submission build: python dev/check_placeholders.py wflo_rydberg.ipynb [more.ipynb ...]
(The notebook cannot check its own markdown without reading its file, which the no-external-files rule forbids.)
"""
import json, re, sys

PATTERNS = [r"DEMO_RUNTIME", r"DEMO_MIN", r"__LITERAL__", r"\bTODO\b", r"\bTBD\b", r"\bXXX\b", r"\bFIXME\b",
            r"\(Phase 5\.\)", r"lorem ipsum", r"\[insert"]
bad = []
for path in sys.argv[1:]:
    nb = json.load(open(path))
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for pat in PATTERNS:
            for m in re.finditer(pat, src, re.I):
                bad.append(f"{path} cell {i}: {pat!r} near {src[max(0, m.start() - 40):m.end() + 40]!r}")
assert not bad, "placeholders remain:\n" + "\n".join(bad)
print(f"no placeholders in {', '.join(sys.argv[1:])}")
