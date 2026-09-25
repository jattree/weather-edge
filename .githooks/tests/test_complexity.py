"""Exercise the gate against real Git indexes and actual Ruff metrics."""
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('complexity', REPO / '.githooks/complexity-check.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

PATH = 'src/example.py'


def source(score, name='legacy', suppression=False):
    comment = '  # noqa: C901' if suppression else ''
    return f'def {name}(x):{comment}\n' + '    if x:\n        x -= 1\n' * (score - 1) + '    return x\n'


class ComplexityGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.command('git', 'init', '-q')
        self.command('git', 'config', 'user.email', 'test@example.invalid')
        self.command('git', 'config', 'user.name', 'Complexity test')
        (self.root / '.githooks').mkdir()
        shutil.copy2(REPO / '.githooks' / 'complexity-check.py', self.root / '.githooks')
        (self.root / 'src').mkdir()
        self.command('git', 'commit', '--allow-empty', '-qm', 'base')

    def command(self, *args):
        return subprocess.run(args, cwd=self.root, text=True, capture_output=True, check=True)

    def write(self, score, **kwargs):
        (self.root / PATH).write_text(source(score, **kwargs))

    def stage(self):
        self.command('git', 'add', PATH)

    def baseline(self, score):
        self.write(score, suppression=True)
        self.stage()
        self.command('git', 'commit', '-qm', 'legacy baseline')

    def check_gate(self, expected, *args):
        result = subprocess.run(['python3', '.githooks/complexity-check.py', *args],
                                cwd=self.root, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_legacy_ceiling_and_suppressions(self):
        self.baseline(16)
        for score, expected in [(16, 0), (14, 0), (17, 1)]:
            self.write(score, suppression=True)
            self.stage()
            self.check_gate(expected, '--cached')

    def test_new_functions_block_above_ten(self):
        for score, expected in [(10, 0), (11, 1)]:
            self.write(score, suppression=True)
            self.stage()
            self.check_gate(expected, '--cached')

    def test_staged_blob_cannot_be_hidden_by_unstaged_fix(self):
        self.baseline(16)
        self.write(17)
        self.stage()
        self.write(10)
        self.check_gate(1, '--cached')
        self.stage()
        self.write(20)
        self.check_gate(0, '--cached')

    def test_reduction_becomes_new_ceiling_and_ci_matches(self):
        self.baseline(16)
        self.baseline(12)
        self.write(13)
        self.stage()
        self.check_gate(1, '--cached')
        self.command('git', 'commit', '-qm', 'regression')
        self.check_gate(1, '--base', 'HEAD^', '--head', 'HEAD')

    def test_syntax_error_blocks(self):
        (self.root / PATH).write_text('def broken(:\n')
        self.stage()
        self.check_gate(2, '--cached')

    def test_qualified_methods_do_not_share_ceilings(self):
        before = [{'key': 'A/run', 'score': 20}, {'key': 'B/run', 'score': 5}]
        after = [{'key': 'A/run', 'score': 20}, {'key': 'B/run', 'score': 11}]
        self.assertEqual(len(gate.compare(before, after)), 1)

    def test_added_ambiguous_callback_cannot_inherit_legacy_ceiling(self):
        before = [{'key': 'effect', 'score': 20}]
        after = [{'key': 'effect', 'score': 20}, {'key': 'effect', 'score': 11}]
        self.assertEqual(len(gate.compare(before, after)), 2)


if __name__ == '__main__':
    unittest.main()
