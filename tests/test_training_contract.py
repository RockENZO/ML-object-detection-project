"""Training orchestration contracts; no real model, GPU or metric evidence."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import performance_study as study
from export_training import export_training
from prepare_grouped_split import digest

ROOT = Path(__file__).resolve().parents[1]


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output, self.run = self.root / 'study', self.root / 'runs'
        self.output.mkdir()
        for name in ['manifest.json', 'augmentation.json']:
            (self.output / name).write_text('{}')
        self.init = self.root / 'custom.pt'
        self.init.write_bytes(b'custom initialization fixture')
        self.saved, self.calls, self.fail = {}, [], False
        self.fail_before_checkpoint = False
        outer = self

        class FakeYOLO:
            def __init__(self, path):
                self.path = path

            def train(self, **options):
                outer.calls.append(options)
                if not options.get('resume'):
                    outer.saved = dict(options)
                if outer.fail_before_checkpoint:
                    raise RuntimeError('failed before first checkpoint fixture')
                weights = outer.run / 'named/weights'
                weights.mkdir(parents=True, exist_ok=True)
                (weights / 'last.pt').write_bytes(b'last checkpoint fixture')
                (weights / 'best.pt').write_bytes(b'best checkpoint fixture')
                if outer.fail:
                    raise RuntimeError('interrupted fixture')

            def val(self, **options):
                return types.SimpleNamespace(
                    box=types.SimpleNamespace(map50=.7, map=.5, mp=.8, mr=.6),
                    summary=lambda: [],
                )

        torch = types.ModuleType('torch'); torch.__version__ = 'fixture'
        ultra = types.ModuleType('ultralytics'); ultra.__version__ = 'fixture'; ultra.__path__ = []; ultra.YOLO = FakeYOLO
        nn = types.ModuleType('ultralytics.nn'); nn.__path__ = []
        tasks = types.ModuleType('ultralytics.nn.tasks')
        tasks.torch_safe_load = lambda path: ({'train_args': self.saved}, path)
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(sys.modules, {'torch': torch, 'ultralytics': ultra,
                                                        'ultralytics.nn': nn, 'ultralytics.nn.tasks': tasks}))
        self.augmentation = self.stack.enter_context(patch.object(study, 'degrade_augmentation', return_value=lambda: None))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def train(self, **options):
        defaults = dict(output=self.output, run=self.run, variant='improved', epochs=5,
                        imgsz=640, device='cpu', name='named')
        defaults.update(options)
        study.train(**defaults)

    def report(self):
        return json.loads((self.run / 'named-selection.json').read_text())

    def test_resume_restores_initial_hash_degradation_and_all_saved_hyp(self):
        initial_hash = digest(self.init)
        self.train(init=self.init, degrade=.4, hyp={'lr0': .002, 'cos_lr': True, 'close_mosaic': 8})
        self.init.unlink()  # A resume must not need or rehash a CLI-default initialization.
        self.train(resume=True)
        report = self.report()
        self.assertEqual(report['initial_checkpoint_sha256'], initial_hash)
        self.assertEqual(report['options']['degrade_probability'], .4)
        for key in ['lr0', 'cos_lr', 'close_mosaic']:
            self.assertEqual(report['options'][key], self.saved[key])
        self.assertEqual([c.args[0] for c in self.augmentation.call_args_list], [.4, .4])
        self.assertEqual(report['resumed_from_checkpoint_sha256'], digest(self.run / 'named/weights/last.pt'))

    def test_interrupted_run_saves_resume_metadata_before_training_finishes(self):
        self.fail = True
        with self.assertRaises(RuntimeError):
            self.train(init=self.init, degrade=.4)
        self.assertTrue((self.run / 'named-training.json').is_file())
        self.assertFalse((self.run / 'named-selection.json').exists())
        self.fail = False
        self.train(resume=True)
        self.assertEqual(self.report()['options']['degrade_probability'], .4)

    def test_resume_restarts_unstarted_run_from_original_settings(self):
        self.fail_before_checkpoint = True
        with self.assertRaises(RuntimeError):
            self.train(init=self.init, degrade=.4, hyp={'lr0': .002})
        self.assertFalse((self.run / 'named/weights/last.pt').exists())
        self.fail_before_checkpoint = False
        self.train(resume=True)
        self.assertTrue(self.report()['restarted_before_first_checkpoint'])
        self.assertIsNone(self.report()['resumed_from_checkpoint_sha256'])
        self.assertEqual(self.calls[-1]['lr0'], .002)
        self.assertEqual(self.report()['options']['degrade_probability'], .4)
        self.assertTrue(self.calls[-1]['exist_ok'])

    def test_resume_rejects_changed_probability_hyp_and_dataset(self):
        self.train(init=self.init, degrade=.4, hyp={'lr0': .002})
        count = len(self.calls)
        for options in [{'degrade': 0}, {'hyp': {'lr0': .01}}, {'init': self.root / 'other.pt'}]:
            (self.root / 'other.pt').write_bytes(b'different initialization')
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.train(resume=True, **options)
        (self.output / 'manifest.json').write_text('{"changed":true}')
        with self.assertRaises(ValueError):
            self.train(resume=True)
        self.assertEqual(len(self.calls), count)

    def test_legacy_resume_without_original_metadata_fails_before_training(self):
        with self.assertRaisesRegex(ValueError, 'metadata'):
            self.train(resume=True)
        self.assertEqual(self.calls, [])

    def test_reserved_hyp_keys_and_non_object_hyp_fail_before_training(self):
        for hyp in [{'name': 'redirect'}, {'project': '/other'}, {'data': 'other.yaml'},
                    {'resume': True}, {'batch': 4}, [], 'not an object']:
            with self.subTest(hyp=hyp), self.assertRaises(ValueError):
                self.train(init=self.init, hyp=hyp)
        self.assertEqual(self.calls, [])

    def test_invalid_degradation_probabilities_fail_before_training(self):
        for value in [-1, 40, float('nan'), float('inf'), True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.train(init=self.init, degrade=value)
        self.assertEqual(self.calls, [])

    def test_competing_fresh_run_cannot_reuse_another_process_metadata(self):
        model_class = sys.modules['ultralytics'].YOLO
        def competing_claim(path):
            self.run.mkdir(parents=True)
            (self.run / 'named-training.json').write_text('{"owner":"other process"}')
            return model_class(path)
        with patch.object(sys.modules['ultralytics'], 'YOLO', side_effect=competing_claim):
            with self.assertRaises(FileExistsError):
                self.train(init=self.init)
        self.assertEqual(self.calls, [])
        self.assertEqual(json.loads((self.run / 'named-training.json').read_text()), {'owner': 'other process'})

    def test_fresh_named_run_cannot_silently_redirect_or_overwrite(self):
        self.train(init=self.init)
        with self.assertRaises(ValueError):
            self.train(init=self.init)
        for name in ['../outside', '/absolute', '..', 'a/b']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.train(init=self.init, name=name)


class ExportContractTests(unittest.TestCase):
    def test_exported_hash_identifies_stripped_bytes_and_preserves_training_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / 'named'; output = root / 'export'
            (run / 'weights').mkdir(parents=True)
            source = run / 'weights/best.pt'; source.write_bytes(b'original checkpoint fixture')
            (run / 'results.csv').write_text('epoch,metric\n1,0\n')
            state = root / 'named-training.json'; state.write_text('{"fixture":true}')
            metadata_bytes = state.read_bytes()
            (root / 'named-selection.json').write_text(json.dumps({
                'checkpoint_sha256': digest(source), 'training_metadata_sha256': digest(state)}))
            def strip(source_path, target_path):
                Path(target_path).write_bytes(Path(source_path).read_bytes() + b':serialized without optimizer')
                state.write_text('{"changed during export":true}')
            report = export_training(run, output, stripper=strip)
            self.assertEqual(report['training_checkpoint_sha256'], digest(source))
            self.assertEqual(report['checkpoint_sha256'], digest(output / 'best.pt'))
            self.assertNotEqual(report['training_checkpoint_sha256'], report['checkpoint_sha256'])
            self.assertEqual(json.loads((output / 'selection.json').read_text()), report)
            self.assertEqual((output / 'training.json').read_bytes(), metadata_bytes)

    def test_changed_or_missing_metadata_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / 'named'; (run / 'weights').mkdir(parents=True)
            source = run / 'weights/best.pt'; source.write_bytes(b'checkpoint')
            state = root / 'named-training.json'; state.write_text('{"original":true}')
            (root / 'named-selection.json').write_text(json.dumps({
                'checkpoint_sha256': digest(source), 'training_metadata_sha256': digest(state)}))
            state.write_text('{"changed":true}')
            for missing in [False, True]:
                if missing: state.unlink()
                with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, 'metadata'):
                    export_training(run, root / 'out', stripper=lambda *a: self.fail('must not strip'))
                self.assertFalse((root / 'out').exists())

    def test_mismatched_source_hash_does_not_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / 'named'; (run / 'weights').mkdir(parents=True)
            (run / 'weights/best.pt').write_bytes(b'changed')
            (root / 'named-selection.json').write_text('{"checkpoint_sha256":"stale"}')
            with self.assertRaises(ValueError):
                export_training(run, root / 'out', stripper=lambda *args: self.fail('must not strip'))
            self.assertFalse((root / 'out').exists())


class ShellContractTests(unittest.TestCase):
    def make_python(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!' + sys.executable + '\n' + '''import sys, os, json
from pathlib import Path
with open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1:3] == ['-m', 'pip'] and os.environ.get('FAIL_ONCE'):
    flag = Path(os.environ['FAIL_ONCE'])
    if not flag.exists(): flag.touch(); sys.exit(1)
''')
        path.chmod(0o755)

    def test_docker_entrypoint_forwards_settings_and_checks_output_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); app = root / 'app'; app.mkdir(); output = root / 'output'
            (app / 'artifacts/performance-study').mkdir(parents=True)
            (app / 'artifacts/performance-study/manifest.json').write_text('{}')
            bin_ = root / 'bin'; self.make_python(bin_ / 'python')
            script = root / 'entrypoint.sh'
            script.write_text((ROOT / 'docker/train_entrypoint.sh').read_text().replace('cd /app', 'cd "' + str(app) + '"').replace('/output/', str(output) + '/'))
            log = root / 'calls.jsonl'
            env = {**os.environ, 'PATH': str(bin_) + os.pathsep + os.defpath, 'CALL_LOG': str(log),
                   'NAME': 'fixture', 'BATCH': '4', 'HYP': '{"lr0":0.002}', 'DEGRADE': '0.4'}
            subprocess.run(['bash', str(script)], env=env, check=True, capture_output=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            train = next(a for a in calls if a[:2] == ['performance_study.py', 'train'])
            for key, value in [('--batch', '4'), ('--hyp', '{"lr0":0.002}'), ('--degrade', '0.4')]:
                self.assertEqual(train[train.index(key) + 1], value)
            shutil.rmtree(output); output.write_text('not a writable output directory'); log.unlink()
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(log.exists(), 'Training must not start before output preflight')

    def test_one_folder_retries_interrupted_install_and_reuses_only_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); repo = root / 'repo'; (repo / 'tools').mkdir(parents=True)
            shutil.copy(ROOT / 'tools/train_in_one_folder.sh', repo / 'tools/train_in_one_folder.sh')
            shutil.copy(ROOT / 'requirements-experiment.txt', repo / 'requirements-experiment.txt')
            work = root / 'work'; self.make_python(work / 'venv/bin/python')
            (work / 'artifacts/performance-study').mkdir(parents=True)
            (work / 'artifacts/performance-study/manifest.json').write_text('{}')
            log = root / 'calls.jsonl'
            env = {**os.environ, 'WORK': str(work), 'CALL_LOG': str(log), 'FAIL_ONCE': str(root / 'failed-once')}
            command = ['bash', str(repo / 'tools/train_in_one_folder.sh')]
            self.assertNotEqual(subprocess.run(command, env=env, capture_output=True).returncode, 0)
            self.assertFalse((work / 'requirements-installed.txt').exists())
            subprocess.run(command, env=env, check=True, capture_output=True)
            subprocess.run(command, env=env, check=True, capture_output=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(sum(c[:2] == ['-m', 'pip'] for c in calls), 2)
            self.assertEqual((work / 'requirements-installed.txt').read_bytes(), (repo / 'requirements-experiment.txt').read_bytes())

    @unittest.skipUnless(shutil.which('docker'), 'Docker CLI needed for Compose expansion test')
    def test_compose_forwards_training_settings_and_never_creates_export_as_root(self):
        env = {**os.environ, 'BATCH': '4', 'HYP': '{"lr0":0.002}', 'DEGRADE': '0.4'}
        raw = subprocess.check_output(['docker', 'compose', '-f', str(ROOT / 'docker-compose.train.yml'), 'config', '--format', 'json'], env=env, text=True)
        service = json.loads(raw)['services']['train']
        for key in ['BATCH', 'HYP', 'DEGRADE']:
            self.assertEqual(service['environment'][key], env[key])
        export = next(v for v in service['volumes'] if v['target'] == '/output')
        # Compose releases may omit false-valued fields from normalized JSON.
        self.assertFalse(export.get('bind', {}).get('create_host_path', False))
        self.assertIn('create_host_path: false', (ROOT / 'docker-compose.train.yml').read_text())


if __name__ == '__main__':
    unittest.main()
