#!/usr/bin/env python3
"""Block new complexity >10 and increases above 10, using Git blobs, not disk.

Pre-commit: --cached. CI: --base <PR base SHA> --head HEAD.
The previous revision is the baseline, so reductions ratchet automatically.
Renamed files/functions are new identities and must meet the default ceiling.

Ported from the nexted course-builder gate, Python only.
"""

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

LIMIT = 10
ROOT = Path(__file__).resolve().parents[1]
# Prefer the project venv's ruff so the hook works without a global install.
RUFF = str(ROOT / '.venv/bin/ruff') if (ROOT / '.venv/bin/ruff').exists() else shutil.which('ruff') or 'ruff'


def run(command, source=None, allowed=(0,)):
    result = subprocess.run(command, input=source, text=True, capture_output=True,
                            cwd=ROOT, timeout=120, check=False)
    if result.returncode not in allowed:
        raise RuntimeError(f"{' '.join(command)}: {result.stderr or result.stdout}")
    return result.stdout


def git(*args):
    return run(['git', *args])


def function_names(source):
    names = {}

    def visit(node, scope):
        nested = scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nested = [*scope, node.name]
            names[node.lineno] = '/'.join(nested)
        for child in ast.iter_child_nodes(node):
            visit(child, nested)

    visit(ast.parse(source), [])
    return names


def python_metrics(file):
    names = function_names(file['source'])
    output = run([RUFF, 'check', '--isolated', '--no-cache', '--ignore-noqa',
                  '--select', 'C901', '--config', 'lint.mccabe.max-complexity=0',
                  '--output-format=json', '--stdin-filename', file['path'], '-'],
                 file['source'], allowed=(0, 1))
    metrics = []
    for finding in json.loads(output):
        if finding['code'] != 'C901':
            raise RuntimeError(f"Cannot measure {file['path']}: {finding['message']}")
        line = finding['location']['row']
        score = int(re.search(r'\((\d+) > 0\)', finding['message'])[1])
        metrics.append({'key': names[line], 'line': line, 'score': score})
    return metrics


def measure(files):
    return [python_metrics(file) for file in files]


def compare(before, after):
    previous = defaultdict(list)
    current = defaultdict(list)
    for item in before:
        previous[item['key']].append(item)
    for item in after:
        current[item['key']].append(item)
    violations = []
    for key, items in current.items():
        old = previous[key]
        # Ambiguous callbacks cannot inherit a ceiling when their count changes.
        ceilings = [item['score'] for item in old] if len(old) == len(items) else []
        for index, item in enumerate(items):
            ceiling = max(LIMIT, ceilings[index]) if ceilings else LIMIT
            if item['score'] > ceiling:
                violations.append((item, ceiling))
    return violations


def snapshots(base, head, cached):
    diff = ['--cached', base] if cached else [base, head]
    paths = git('diff', '--no-renames', '--name-only', '-z', '--diff-filter=ACM',
                *diff).split('\0')
    existing = set(git('ls-tree', '-r', '--name-only', '-z', base).split('\0'))
    pairs = []
    for path in paths:
        if Path(path).suffix != '.py':
            continue
        before = git('show', f'{base}:{path}') if path in existing else ''
        after = git('show', f':{path}' if cached else f'{head}:{path}')
        pairs.append(({'path': path, 'source': before}, {'path': path, 'source': after}))
    return pairs


def check(base, head, cached):
    pairs = snapshots(base, head, cached)
    if not pairs:
        return 0
    before = measure([pair[0] for pair in pairs])
    after = measure([pair[1] for pair in pairs])
    count = 0
    for pair, old, new in zip(pairs, before, after, strict=True):
        for item, ceiling in compare(old, new):
            print(f"{pair[1]['path']}:{item['line']}: {item['key']} "
                  f"complexity {item['score']} exceeds {ceiling}")
            count += 1
    print(f'complexity: checked {len(pairs)} changed files; {count} complexity regressions')
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cached', action='store_true')
    mode.add_argument('--base')
    parser.add_argument('--head', default='HEAD')
    args = parser.parse_args()
    return int(bool(check(args.base or 'HEAD', args.head, args.cached)))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, SyntaxError, ValueError, subprocess.TimeoutExpired) as error:
        print(f'Complexity check failed: {error}', file=sys.stderr)
        sys.exit(2)
