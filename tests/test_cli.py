"""Dependency-free checks for the inference/evaluation command wiring."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

import yolo_inference


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "weights.pt"
        self.model.touch()
        self.video = self.root / "video.mp4"
        self.video.touch()
        self.data = self.root / "data.yaml"
        self.data.write_text("test: images\n", encoding="utf-8")
        self.yolo = Mock()
        self.yolo.return_value.val.return_value.box = types.SimpleNamespace(
            mp=0.7, mr=0.8, map50=0.75, map=0.6)
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
            source=str(self.video), save=True, project=str(output.parent),
            name=output.name, exist_ok=True)

    def test_evaluate_records_test_metrics(self):
        output = self.root / "results" / "evaluate"
        self.assertEqual(0, yolo_inference.main([
            "--model", str(self.model), "evaluate", "--data", str(self.data),
            "--output-dir", str(output)]))
        self.yolo.return_value.val.assert_called_once_with(
            data=str(self.data), split="test", imgsz=640, plots=True,
            project=str(output.parent), name=output.name, exist_ok=True)
        report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(report["box_precision"], 0.7)
        self.assertEqual(report["split"], "test")


if __name__ == "__main__":
    unittest.main()
