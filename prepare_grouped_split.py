"""Audit football data and partition filename-derived video groups deterministically."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'training/football-players-detection-1/football-players-detection-1'
NAMES = ['ball','goalkeeper','player','referee']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(root):
    records = []
    for split in ('train','valid','test'):
        for image in sorted((Path(root)/split/'images').glob('*')):
            if image.suffix.lower() not in ('.jpg','.jpeg','.png'):
                continue
            match = re.fullmatch(r'([0-9a-f]+)_\d+_\d+_png\.rf\.[0-9a-f]+', image.stem)
            if not match:
                raise ValueError('Cannot infer video prefix from ' + image.name)
            label = Path(root)/split/'labels'/(image.stem+'.txt')
            if not label.is_file():
                raise ValueError('Missing label ' + str(label))
            classes = []
            for line in label.read_text().splitlines():
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError('Malformed annotation ' + str(label))
                klass = int(fields[0]); box = list(map(float,fields[1:]))
                if klass not in range(4) or not all(0 <= v <= 1 for v in box) or box[2] <= 0 or box[3] <= 0:
                    raise ValueError('Invalid class/box in ' + str(label))
                classes.append(klass)
            records.append({'image': image.relative_to(root).as_posix(), 'label': label.relative_to(root).as_posix(),
                            'image_sha256':digest(image),'label_sha256':digest(label),
                            'group':match[1], 'original_split':split, 'classes':classes})
    if not records:
        raise ValueError('No images found')
    return records


def prepare(root, output, seed=42):
    root,output=Path(root),Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')
    records=collect(root)
    groups=sorted({r['group'] for r in records})
    if len(groups)<3:
        raise ValueError('Need at least three video groups')
    random.Random(seed).shuffle(groups)
    ntest=max(1,round(len(groups)*.2));nval=max(1,round(len(groups)*.2))
    assignment={group: ('test' if i<ntest else 'val' if i<ntest+nval else 'train') for i,group in enumerate(groups)}
    hashes={}
    for r in records:
        previous=hashes.setdefault(r['image_sha256'],r['group'])
        if previous!=r['group']:
            raise ValueError('Identical images in different groups; reconcile group identities before splitting')
        r['split']=assignment[r['group']]
    overlaps={}
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        overlaps[a+'_'+b]=sorted({r['group'] for r in records if r['original_split']==a}&{r['group'] for r in records if r['original_split']==b})
    output.mkdir(parents=True,exist_ok=True)
    for r in records:
        for kind, key in [('images','image'),('labels','label')]:
            target=output/r['split']/kind/Path(r[key]).name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(root/r[key],target)
    # Absolute path is generated at execution time, not checked in.
    (output/'data.yaml').write_text('path: '+json.dumps(str(output.resolve()))+'\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 4\nnames: '+json.dumps(NAMES)+'\n')
    manifest={'seed':seed,'group_rule':'filename hex prefix before frame indices; provisional video identity',
              'names':NAMES,'groups':dict(sorted(assignment.items())),
              'original_split_counts':dict(Counter(r['original_split'] for r in records)),
              'group_overlap_in_original_splits':overlaps,
              'new_split_counts':dict(Counter(r['split'] for r in records)),
              'class_annotation_counts':{split:dict(Counter(str(c) for r in records if r['split']==split for c in r['classes'])) for split in ('train','val','test')},
              'records':records,
              'limitations':['Video identities are inferred from export filenames and need upstream confirmation.',
                             'Different video IDs may still depict the same match/scene; near duplicates are not audited.']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=DATA)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/grouped')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    result=prepare(args.input_dir,args.output_dir,args.seed)
    print(json.dumps({key:result[key] for key in ['original_split_counts','new_split_counts','group_overlap_in_original_splits']},indent=2))
