"""No models are loaded; tiny fixture bytes exercise publication safety."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import fetch_weights as weights


class FetchWeightsTests(unittest.TestCase):
    payload = b'fake checkpoint bytes, never a runnable model'

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'model.pt'

    def download(self, payload=None):
        return patch.object(weights.urllib.request, 'urlopen', return_value=io.BytesIO(self.payload if payload is None else payload))

    def digest(self):
        return patch.object(weights, 'CHECKPOINT_SHA256', hashlib.sha256(self.payload).hexdigest())

    def test_verified_download_publishes_exact_bytes_and_provenance(self):
        with self.download() as get, self.digest():
            report = weights.fetch_weights(self.output)
        self.assertEqual(self.output.read_bytes(), self.payload)
        self.assertEqual(report['bytes'], len(self.payload))
        self.assertEqual(report['source_commit'], weights.SOURCE_COMMIT)
        self.assertFalse(report['model_loaded'])
        get.assert_called_once_with(weights.SOURCE_URL, timeout=30)
        self.assertEqual(list(self.root.iterdir()), [self.output])

    def test_checksum_mismatch_and_lfs_pointer_are_not_published(self):
        for payload in [b'wrong bytes', b'version https://git-lfs.github.com/spec/v1\n', b'']:
            with self.subTest(payload=payload), self.download(payload), self.digest():
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    weights.fetch_weights(self.output)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_oversized_download_is_bounded_and_removed(self):
        with self.download(b'x' * 50), patch.object(weights, 'MAX_BYTES', 8):
            with self.assertRaisesRegex(ValueError, 'limit'):
                weights.fetch_weights(self.output)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_file_directory_and_dangling_symlink_are_preserved(self):
        for kind in ['file', 'directory', 'symlink']:
            with self.subTest(kind=kind):
                if kind == 'file':
                    self.output.write_bytes(b'keep')
                elif kind == 'directory':
                    self.output.mkdir()
                else:
                    self.output.symlink_to(self.root / 'missing')
                with patch.object(weights.urllib.request, 'urlopen') as get:
                    with self.assertRaises(FileExistsError):
                        weights.fetch_weights(self.output)
                    get.assert_not_called()
                if kind == 'directory':
                    self.output.rmdir()
                else:
                    if kind == 'file':
                        self.assertEqual(self.output.read_bytes(), b'keep')
                    else:
                        self.assertTrue(self.output.is_symlink())
                    self.output.unlink()

    def test_concurrent_writer_is_never_overwritten(self):
        link = os.link
        def racing_link(source, destination):
            Path(destination).write_bytes(b'concurrent writer')
            return link(source, destination)
        with self.download(), self.digest(), patch.object(weights.os, 'link', side_effect=racing_link):
            with self.assertRaises(FileExistsError):
                weights.fetch_weights(self.output)
        self.assertEqual(self.output.read_bytes(), b'concurrent writer')
        self.assertEqual(list(self.root.iterdir()), [self.output])

    def test_network_failure_removes_staging_file(self):
        with patch.object(weights.urllib.request, 'urlopen', side_effect=TimeoutError('fixture timeout')):
            with self.assertRaises(TimeoutError):
                weights.fetch_weights(self.output)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_pinned_hash_matches_checked_in_validation_report(self):
        report = Path(__file__).resolve().parents[1] / 'reports/detector_ball1280robust_val_cpu_20261019.json'
        self.assertEqual(json.loads(report.read_text())['checkpoint_sha256'], weights.CHECKPOINT_SHA256)
        self.assertIn('/' + weights.SOURCE_COMMIT + '/', weights.SOURCE_URL)
