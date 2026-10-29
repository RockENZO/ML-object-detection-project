"""Small files verify input contracts, not detector accuracy or image decoding."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from evaluation_contract import verify_test_inputs
from prepare_grouped_split import digest


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name in ('images', 'labels'):
            (self.root/'test'/name).mkdir(parents=True)
        image=self.root/'test/images/frame.jpg';image.write_bytes(b'synthetic image bytes')
        label=self.root/'test/labels/frame.txt';label.write_text('0 .5 .5 .1 .1\n')
        self.dataset={'path':str(self.root), 'test':'test/images'}
        self.split={'records':[dict(split='test',group='held-out',image='source/images/frame.jpg',
            label='source/labels/frame.txt',image_sha256=digest(image),label_sha256=digest(label))]}

    def test_valid_handoff_and_changed_or_missing_artifacts(self):
        self.assertEqual(verify_test_inputs(self.dataset,self.split),self.split['records'])
        image=self.root/'test/images/frame.jpg'
        image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'differs'):
            verify_test_inputs(self.dataset,self.split)
        image.unlink()
        with self.assertRaisesRegex(ValueError,'differs'):
            verify_test_inputs(self.dataset,self.split)

    def test_unlisted_label_directory_and_symlink_are_rejected(self):
        for kind in ('extra-label','nested-directory','symlink'):
            with self.subTest(kind=kind):
                if kind=='extra-label':
                    extra=self.root/'test/labels/extra.txt';extra.write_text('unlisted')
                elif kind=='nested-directory':
                    extra=self.root/'test/images/nested';extra.mkdir()
                else:
                    extra=self.root/'test/images/alias.jpg';extra.symlink_to(self.root/'test/images/frame.jpg')
                with self.assertRaisesRegex(ValueError,'membership'):
                    verify_test_inputs(self.dataset,self.split)
                if extra.is_dir():extra.rmdir()
                else:extra.unlink()

    def test_test_group_cannot_appear_in_training_or_validation(self):
        for split in ('train','val'):
            manifest=deepcopy(self.split)
            manifest['records'].append(dict(split=split,group='held-out'))
            with self.assertRaisesRegex(ValueError,'overlap'):
                verify_test_inputs(self.dataset,manifest)

    def test_relative_root_and_invalid_manifest_are_explicit_errors(self):
        with self.assertRaisesRegex(ValueError,'absolute'):
            verify_test_inputs(dict(path='relative',test='test/images'),self.split)
        for manifest in ({}, {'records':[]}, {'records':[None]}):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                verify_test_inputs(self.dataset,manifest)
