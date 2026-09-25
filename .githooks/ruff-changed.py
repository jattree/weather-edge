#!/usr/bin/env python3
"""Run ruff but report only findings that land on lines this change touched.

Whole-file linting is unusable on a codebase with a large pre-existing finding
backlog: touching any line in a file drags in every unrelated violation already
in it, so the hook either blocks unrelated work or gets bypassed. This filters
ruff's output down to the lines actually added or modified.

  ruff-changed.py --cached                    # staged changes (pre-commit)
  ruff-changed.py --range @{upstream}..HEAD   # outgoing commits (pre-push)
  ruff-changed.py --cached --select F821,E722

Caveat worth knowing: ruff reads the working tree, while --cached diffs the
index. For a partially staged file the reported lines can drift. The script
warns when it sees one rather than pretending otherwise.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

HUNK = "@@"


def sh(*args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode > 1:
        sys.stderr.write(r.stderr)
        raise SystemExit(2)
    return r


def changed_paths(diff_args):
    """Paths added/copied/modified/renamed by this change.

    Uses ``--name-only -z`` rather than parsing the ``+++ b/...`` header. With
    core.quotepath enabled git C-quotes non-ASCII headers, e.g.
    ``+++ "b/\\303\\274.py"``, which then no longer matches the filename ruff
    reports -- so every finding in that file was silently dropped. ``-z`` emits
    raw NUL-separated paths with no quoting.

    R is included: under ACM a rename carrying a new violation was filtered out
    entirely and the whole file went unchecked.
    """
    out = sh("git", "diff", "--name-only", "-z", "--diff-filter=ACMR", *diff_args).stdout
    return [p for p in out.split("\0") if p]


def changed_lines(diff_args):
    """Map {path: set(line numbers added/modified)} from a -U0 diff."""
    result = defaultdict(set)
    for path in changed_paths(diff_args):
        out = sh("git", "diff", "-U0", *diff_args, "--", path).stdout
        for line in out.splitlines():
            if not line.startswith(HUNK):
                continue
            # @@ -old,count +new,count @@
            new = line.split("+", 1)[1].split(HUNK)[0].strip()
            start, _, count = new.partition(",")
            start, count = int(start), int(count or 1)
            result[path].update(range(start, start + count))
    return result


def ruff_binary():
    """The project venv's ruff if present, else whatever is on PATH."""
    root = sh("git", "rev-parse", "--show-toplevel").stdout.strip()
    venv = os.path.join(root, ".venv", "bin", "ruff")
    return venv if os.path.exists(venv) else "ruff"


def ruff_command(args):
    cmd = [ruff_binary(), "check", "--no-cache", "--force-exclude", "--output-format=json"]
    if args.select:
        cmd += ["--select", args.select]
    if args.ignore:
        cmd += ["--ignore", args.ignore]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cached", action="store_true")
    g.add_argument("--range")
    ap.add_argument("--select", help="passed through to ruff --select")
    ap.add_argument("--ignore", help="passed through to ruff --ignore")
    args = ap.parse_args()

    diff_args = ["--cached"] if args.cached else [args.range]
    touched = changed_lines(diff_args)
    py = {p: v for p, v in touched.items() if p.endswith(".py") and os.path.exists(p)}
    if not py:
        return 0

    if args.cached:
        dirty = [p for p in py if sh("git", "diff", "--quiet", "--", p, check=False).returncode]
        if dirty:
            print(f"  ! working tree differs from index; line mapping may drift: {' '.join(dirty)}")

    r = sh(*ruff_command(args), *sorted(py))
    if not r.stdout.strip():
        return 0

    root = sh("git", "rev-parse", "--show-toplevel").stdout.strip()
    hits = []
    for f in json.loads(r.stdout):
        rel = os.path.relpath(f["filename"], root)
        if f["location"]["row"] in py.get(rel, ()):
            hits.append(f)

    for f in sorted(hits, key=lambda x: (x["filename"], x["location"]["row"])):
        rel = os.path.relpath(f["filename"], root)
        loc = f["location"]
        print(f"  {rel}:{loc['row']}:{loc['column']}: {f['code']} {f['message']}")

    if hits:
        n = len(hits)
        print(f"\n  {n} finding{'s' if n != 1 else ''} on changed lines.")
        print("  Pre-existing findings elsewhere in these files are not reported.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
