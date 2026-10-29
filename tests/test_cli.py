"""Dependency-free checks for the inference/evaluation command wiring."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yolo_inference


class CliTests(unittest.TestCase):
    def setUp(self):
        runtime = patch("importlib.metadata.version", return_value="stub-runtime")
        runtime.start()
        self.addCleanup(runtime.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "weights.pt"
        self.model.touch()
        self.video = self.root / "video.mp4"
        self.video.touch()
        self.data = self.root / "data.yaml"
        self.data.write_text(json.dumps({'path':str(self.root), 'names':['ball','goalkeeper','player','referee'], 'test':'test/images'}), encoding="utf-8")
        from prepare_grouped_split import digest
        (self.root/'test/images').mkdir(parents=True)
        (self.root/'test/labels').mkdir(parents=True)
        image=self.root/'test/images/test.jpg'; image.write_bytes(b'test')
        label=self.root/'test/labels/test.txt';label.write_text('0 .5 .5 .1 .1')
        (self.root/'manifest.json').write_text(json.dumps({'records':[dict(split='test',group='group',image='test.jpg',label='test.txt',image_sha256=digest(image),label_sha256=digest(label))]}))
        self.yolo = Mock()
        self.yolo.return_value.val.return_value.summary.return_value = []
        self.yolo.return_value.names = {0:"ball",1:"goalkeeper",2:"player",3:"referee"}
        self.yolo.return_value.val.return_value.box = types.SimpleNamespace(
            mp=0.7, mr=0.8, map50=0.75, map=0.6)
        self.yolo.return_value.val.return_value.speed = {}
        original = sys.modules.get("ultralytics")
        sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=self.yolo)
        self.addCleanup(lambda: sys.modules.pop("ultralytics", None)
                        if original is None else sys.modules.__setitem__("ultralytics", original))

    def test_predict_saves_annotated_output(self):
        output = self.root / "results" / "predict"
        self.assertEqual(0, yolo_inference.main([
            "--model", str(self.model), "predict", "--input", str(self.video),
            "--output-dir", str(output)]))
        self.yolo.return_value.predict.assert_called_once_with(
            source=str(self.video), save=True, imgsz=640, device="cpu", project=str(output.parent),
            name=output.name, exist_ok=True)

    @patch("importlib.metadata.version", return_value="test-runtime")
    def test_evaluate_records_test_metrics(self, _version):
        output = self.root / "results" / "evaluate"
        self.assertEqual(0, yolo_inference.main([
            "--model", str(self.model), "evaluate", "--data", str(self.data),
            "--output-dir", str(output)]))
        self.yolo.return_value.val.assert_called_once_with(
            data=str(self.data), split="test", imgsz=640, plots=True, batch=4, device="cpu",
            project=str(output.parent), name=output.name, exist_ok=True)
        report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(report["box_precision"], 0.7)
        self.assertEqual(report["split"], "test")

    def test_yaml_cannot_redirect_evaluation_away_from_verified_test_files(self):
        for test_path in ['train/images', '/other/images', ['test/images'], 'test.txt', None]:
            with self.subTest(test_path=test_path):
                dataset=json.loads(self.data.read_text())
                dataset['test']=test_path
                self.data.write_text(json.dumps(dataset))
                with self.assertRaisesRegex(SystemExit, 'test/images'):
                    yolo_inference.main(['--model', str(self.model), 'evaluate', '--data', str(self.data), '--output-dir', str(self.root/'out')])
                self.yolo.return_value.val.assert_not_called()

    def test_extra_test_image_is_rejected_before_model_evaluation(self):
        (self.root/'test/images/unreviewed.jpg').write_bytes(b'unreviewed')
        with self.assertRaisesRegex(SystemExit, 'manifest'):
            yolo_inference.main(['--model', str(self.model), 'evaluate', '--data', str(self.data), '--output-dir', str(self.root/'out')])
        self.yolo.return_value.val.assert_not_called()

    def test_duplicate_manifest_record_cannot_inflate_test_count(self):
        path=self.root/'manifest.json'
        manifest=json.loads(path.read_text());manifest['records']*=2
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(SystemExit, 'Duplicate'):
            yolo_inference.main(['--model', str(self.model), 'evaluate', '--data', str(self.data), '--output-dir', str(self.root/'out')])
        self.yolo.return_value.val.assert_not_called()

    def test_label_name_must_match_what_the_image_loader_uses(self):
        path=self.root/'manifest.json'
        manifest=json.loads(path.read_text())
        (self.root/'test/labels/test.txt').rename(self.root/'test/labels/other.txt')
        manifest['records'][0]['label']='other.txt';path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(SystemExit, 'label'):
            yolo_inference.main(['--model', str(self.model), 'evaluate', '--data', str(self.data), '--output-dir', str(self.root/'out')])
        self.yolo.return_value.val.assert_not_called()

    def test_coco_checkpoint_rejected(self):
        self.yolo.return_value.names = {0: 'person', 1: 'bicycle'}
        with self.assertRaisesRegex(SystemExit, 'incompatible'):
            yolo_inference.main(['--model', str(self.model), 'evaluate', '--data', str(self.data), '--output-dir', str(self.root/'out')])
        self.yolo.return_value.val.assert_not_called()


if __name__ == "__main__":
    unittest.main()
