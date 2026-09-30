import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from performance_study import prepare,augment
from finalize_study import validate_role
from prepare_grouped_split import digest

class StudyTests(unittest.TestCase):
    def test_freezes_new_holdout_and_quarantines_previous_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'artifacts/grouped').mkdir(parents=True)
            groups={'aa':'train','bb':'train','cc':'train','dd':'train','ee':'val','ff':'test'}
            (root/'artifacts/grouped/manifest.json').write_text(json.dumps({'groups':groups}))
            data=root/'source';records=[]
            for group in groups:
                image=data/'train/images'/(group+'.jpg');label=data/'train/labels'/(group+'.txt')
                image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
                image.write_bytes(group.encode());label.write_text('0 .5 .5 .1 .1')
                records.append(dict(image=image.relative_to(data).as_posix(),label=label.relative_to(data).as_posix(),group=group,classes=[0]))
            with patch('performance_study.ROOT',root),patch('performance_study.DATA',data),patch('performance_study.collect',return_value=records):
                report=prepare(root/'study')
            self.assertEqual(report['groups']['ff'],'quarantine')
            self.assertEqual(report['groups']['ee'],'val')
            self.assertEqual(len(report['final_groups']),2)
            self.assertTrue(all(groups[group]=='train' for group in report['final_groups']))
            self.assertFalse((root/'study/quarantine').exists())

    def test_frozen_artifact_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            image=root/'val/images/frame.jpg';label=root/'val/labels/frame.txt'
            image.parent.mkdir(parents=True);label.parent.mkdir(parents=True)
            image.write_bytes(b'frame');label.write_text('0 .5 .5 .1 .1')
            (root/'manifest.json').write_text(json.dumps({'records':[{'split':'val','image':'frame.jpg','label':'frame.txt','image_sha256':digest(image),'label_sha256':digest(label)}]}))
            validate_role(root,'val')
            label.write_text('0 .5 .5 .2 .2')
            with self.assertRaisesRegex(ValueError,'Split artifact changed'):
                validate_role(root,'val')

    @unittest.skipUnless(importlib.util.find_spec('PIL'),'Pillow needed for crop test')
    def test_ball_crop_is_training_only_and_normalized(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);images=root/'train/images';labels=root/'train/labels';images.mkdir(parents=True);labels.mkdir(parents=True)
            Image.new('RGB',(1200,900)).save(images/'frame.jpg')
            (labels/'frame.txt').write_text('0 .5 .5 .01 .01\n2 .55 .5 .05 .2\n')
            self.assertEqual(augment(root),1)
            self.assertFalse((root/'val').exists())
            for label in (root/'augmented/labels').glob('*ballcrop.txt'):
                rows=[line.split() for line in label.read_text().splitlines()]
                self.assertTrue(any(row[0]=='0' for row in rows))
                self.assertTrue(all(0<=float(value)<=1 for row in rows for value in row[1:]))

if __name__=='__main__':unittest.main()
