import json
from pathlib import Path
import tempfile
import unittest
from prepare_grouped_split import prepare, collect

class SplitTests(unittest.TestCase):
    def fixture(self, root):
        for split in ['train','valid','test']:
            for group in ['aa','bb','cc','dd','ee']:
                name=f'{group}_0_1_png.rf.{group}'
                for kind,suffix,data in [('images','.jpg',group),('labels','.txt','0 0.5 0.5 0.1 0.1')]:
                    target=root/split/kind/(name+suffix);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(data)

    def test_group_assignment_and_repeatability(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input';self.fixture(source)
            a=prepare(source,root/'a');b=prepare(source,root/'b')
            self.assertEqual(a,b)
            for group in a['groups']:
                self.assertEqual(len({r['split'] for r in a['records'] if r['group']==group}),1)
            for left,right in [('train','val'),('train','test'),('val','test')]:
                self.assertFalse({r['image_sha256'] for r in a['records'] if r['split']==left}&{r['image_sha256'] for r in a['records'] if r['split']==right})

    def test_malformed_annotation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            next((root/'train/labels').glob('*')).write_text('9 .5 .5 .1 .1')
            with self.assertRaises(ValueError):collect(root)

if __name__=='__main__':unittest.main()
