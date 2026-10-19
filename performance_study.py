"""Validation-driven detector improvement on a newly reserved video-prefix holdout."""
import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path
import random
import shutil
import time
from prepare_grouped_split import DATA, ROOT, NAMES, collect, digest


def prepare(output, grouped=None):
    if output.exists():
        raise ValueError('Study output already exists')
    grouped=Path(grouped) if grouped else ROOT/'artifacts/grouped'
    previous=json.loads((grouped/'manifest.json').read_text())
    candidates=sorted(g for g,s in previous['groups'].items() if s=='train')
    random.Random(20260930).shuffle(candidates)
    final=set(candidates[:2])
    records=collect(DATA)
    output.mkdir(parents=True)
    for record in records:
        original=previous['groups'][record['group']]
        record['split']='test' if record['group'] in final else 'quarantine' if original=='test' else original
        if record['split']=='quarantine':continue
        for kind,key in [('images','image'),('labels','label')]:
            target=output/record['split']/kind/Path(record[key]).name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(DATA/record[key],target)
    manifest={'protocol':'new final holdout from former training groups; prior tested groups quarantined',
              'seed':20260930,'previous_manifest_sha256':digest(grouped/'manifest.json'),
              'names':NAMES,'records':records,'counts':dict(Counter(r['split'] for r in records)),
              'final_groups':sorted(final),'groups':{g:next(r['split'] for r in records if r['group']==g) for g in previous['groups']},
              'selection':'Use validation only; train fresh from COCO, never former trained weights',
              'limitations':['Filename-prefix video identities remain provisional; distinct IDs may share scenes.']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    for name in ['base','improved']:
        (output/(name+'.yaml')).write_text('path: '+json.dumps(str(output.resolve()))+'\ntrain: '+('train/images' if name=='base' else 'augmented/images')+'\nval: val/images\ntest: test/images\nnc: 4\nnames: '+json.dumps(NAMES)+'\n')
    return manifest


def augment(output):
    """Native-resolution ball-centred crops supplement full-frame training images only."""
    from PIL import Image
    rng=random.Random(42)
    images=output/'augmented/images';labels=output/'augmented/labels'
    images.mkdir(parents=True);labels.mkdir(parents=True)
    crop_manifest=[]
    for image in sorted((output/'train/images').glob('*.jpg')):
        label=output/'train/labels'/(image.stem+'.txt')
        shutil.copy2(image,images/image.name);shutil.copy2(label,labels/label.name)
        boxes=[list(map(float,line.split())) for line in label.read_text().splitlines()]
        balls=[b for b in boxes if int(b[0])==0]
        if not balls:continue
        with Image.open(image) as frame:
            width,height=frame.size;cw,ch=min(640,width),min(640,height)
            ball=balls[0];left=max(0,min(width-cw,round(ball[1]*width-cw/2+rng.uniform(-.2,.2)*cw)))
            top=max(0,min(height-ch,round(ball[2]*height-ch/2+rng.uniform(-.2,.2)*ch)))
            cropped=[]
            for klass,cx,cy,bw,bh in boxes:
                x1,y1=(cx-bw/2)*width,(cy-bh/2)*height;x2,y2=(cx+bw/2)*width,(cy+bh/2)*height
                a,b,c,d=max(x1,left),max(y1,top),min(x2,left+cw),min(y2,top+ch)
                # Keep objects whose centers lie in crop and at least 70% area is visible.
                if c>a and d>b and left<=cx*width<left+cw and top<=cy*height<top+ch and (c-a)*(d-b)/((x2-x1)*(y2-y1))>=.7:
                    cropped.append(f'{int(klass)} {(a+c-2*left)/(2*cw):.8f} {(b+d-2*top)/(2*ch):.8f} {(c-a)/cw:.8f} {(d-b)/ch:.8f}')
            if not any(line.startswith('0 ') for line in cropped):continue
            name=image.stem+'_ballcrop'
            frame.crop((left,top,left+cw,top+ch)).save(images/(name+'.jpg'),quality=95)
            (labels/(name+'.txt')).write_text('\n'.join(cropped)+'\n')
            crop_manifest.append({'source':image.name,'crop':[left,top,cw,ch],'image_sha256':digest(images/(name+'.jpg')),'label_sha256':digest(labels/(name+'.txt'))})
    (output/'augmentation.json').write_text(json.dumps({'policy':'one native 640-pixel ball-centred crop per eligible training frame; retain full frame; >=70% visibility','seed':42,'crops':crop_manifest},indent=2,sort_keys=True)+'\n')
    return len(crop_manifest)


def degrade_augmentation(probability=0.4, seed=42):
    """Randomly blur, add noise or JPEG-compress training images (labels unchanged) to harden the detector.

    Ultralytics' built-in blur transforms need albumentations and fire 1% of the time, so this wraps the same
    pipeline step with OpenCV. It is applied after mosaic/affine, to training batches only.
    """
    import cv2
    import numpy as np
    from ultralytics.data import augment
    rng=np.random.default_rng(seed)
    original=augment.Albumentations.__call__

    def degraded(self, labels):
        labels=original(self, labels)
        if rng.random()>=probability:
            return labels
        image=labels['img']
        for _ in range(int(rng.integers(1, 3))):
            kind=int(rng.integers(0, 4))
            if kind==0:
                image=cv2.GaussianBlur(image, (0, 0), float(rng.uniform(0.8, 2.5)))
            elif kind==1:  # motion blur along a random direction
                size=int(rng.integers(3, 8))*2+1
                kernel=np.zeros((size, size), np.float32)
                kernel[size//2, :]=1
                matrix=cv2.getRotationMatrix2D((size/2-.5, size/2-.5), float(rng.uniform(0, 180)), 1)
                kernel=cv2.warpAffine(kernel, matrix, (size, size))
                image=cv2.filter2D(image, -1, kernel/max(kernel.sum(), 1e-6))
            elif kind==2:
                noisy=image.astype(np.float32)+rng.normal(0, float(rng.uniform(4, 14)), image.shape)
                image=np.clip(noisy, 0, 255).astype(np.uint8)
            else:
                ok, encoded=cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 60))])
                if ok:
                    image=cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        labels['img']=np.ascontiguousarray(image)
        return labels

    augment.Albumentations.__call__=degraded
    return lambda: setattr(augment.Albumentations, "__call__", original)


# HYP may tune learning/augmentation, but must not redirect this runner's artifacts.
RESERVED_TRAIN_KEYS = {
    'data', 'project', 'name', 'save_dir', 'model', 'resume', 'pretrained', 'cfg',
    'task', 'mode', 'exist_ok', 'split', 'workers', 'seed', 'deterministic',
    'device', 'batch', 'epochs', 'imgsz',
}


def probability(value):
    if isinstance(value, bool):
        raise ValueError('Degradation probability must be finite and between 0 and 1')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Degradation probability must be finite and between 0 and 1')
    return value


def training_state(run, name, split_hash, augmentation_hash, resume, init, degrade):
    path = run / (name + '-training.json')
    if resume:
        if not path.is_file():
            raise ValueError('Resume requires the original training metadata sidecar: ' + str(path)
                             + '; preserve this run and start a new named run rather than guessing its settings')
        state = json.loads(path.read_text())
        if state.get('schema_version') != 1 or not re.fullmatch(r'[0-9a-f]{64}', state.get('initial_checkpoint_sha256', '')):
            raise ValueError('Invalid training metadata')
        if state.get('split_manifest_sha256') != split_hash or state.get('augmentation_sha256') != augmentation_hash:
            raise ValueError('Resume dataset provenance differs from the original run')
        stored_probability = probability(state['degrade_probability'])
        if degrade is not None and probability(degrade) != stored_probability:
            raise ValueError('Resume must preserve the original degradation probability')
        if init is not None and digest(Path(init)) != state['initial_checkpoint_sha256']:
            raise ValueError('Resume must preserve the original initialization hash')
        return path, state
    if path.exists() or (run / name).exists():
        raise ValueError('Run already exists; use --resume or choose a new name')
    return path, {
        'schema_version': 1,
        'initial_checkpoint_sha256': digest(init),
        'split_manifest_sha256': split_hash,
        'augmentation_sha256': augmentation_hash,
        'degrade_probability': probability(0.0 if degrade is None else degrade),
        'augmentation_seed': 42,
    }


def train(output, run, variant, epochs, imgsz, device="mps", name=None, resume=False, init=None, batch=8, hyp=None, degrade=None):
    output, run = Path(output).resolve(), Path(run).resolve()
    name = name or variant
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Run name must be a single safe directory name')
    if hyp is not None and (not isinstance(hyp, dict) or not all(isinstance(k, str) for k in hyp)):
        raise ValueError('HYP must be a JSON object of training hyperparameters')
    reserved = RESERVED_TRAIN_KEYS.intersection(hyp or {})
    if reserved:
        raise ValueError('HYP cannot override runner keys: ' + ', '.join(sorted(reserved)))
    if degrade is not None:
        degrade = probability(degrade)
    init = Path(init) if init is not None else None
    fresh_init = init or ROOT / 'models/yolov8s.pt'
    split_hash = digest(output / 'manifest.json')
    augmentation_hash = digest(output / 'augmentation.json') if variant == 'improved' else None
    state_path, state = training_state(run, name, split_hash, augmentation_hash, resume,
                                       init if resume else fresh_init, degrade)
    import platform
    import torch
    import ultralytics
    from ultralytics import YOLO
    options = dict(data=str(output / (variant + '.yaml')), epochs=epochs, imgsz=imgsz, batch=batch,
                   seed=42, deterministic=True, device=device, workers=0, patience=20,
                   project=str(run), name=name, exist_ok=False, plots=True)
    options.update(hyp or {})
    resumed_hash = None
    if resume:
        from ultralytics.nn.tasks import torch_safe_load
        last = run / name / 'weights/last.pt'
        resumed_hash = digest(last)
        saved = dict(torch_safe_load(str(last))[0]['train_args'])
        if (Path(saved['project']).resolve() != run or saved['name'] != name
                or Path(saved['data']).resolve() != output / (variant + '.yaml')):
            raise ValueError('Checkpoint paths do not match this run and dataset; do not redirect a resumed run')
        if any(key not in saved or saved[key] != value for key, value in (hyp or {}).items()):
            raise ValueError('Resume HYP differs from the checkpoint settings')
        options = {**saved, 'resume': True}
        model = YOLO(str(last))
    else:
        model = YOLO(str(fresh_init))
    run.mkdir(parents=True, exist_ok=True)
    if not state_path.exists():
        # Save non-Ultralytics settings before training, including interrupted runs.
        with state_path.open('x') as handle:
            handle.write(json.dumps(state, indent=2) + '\n')
    restore = degrade_augmentation(state['degrade_probability'], state['augmentation_seed']) if state['degrade_probability'] else lambda: None
    start = time.monotonic()
    try:
        if resume:
            model.train(resume=True)
        else:
            model.train(**options)
    finally:
        restore()
    checkpoint = run / name / 'weights/best.pt'
    selected = YOLO(str(checkpoint))
    metrics = selected.val(data=options['data'], split='val', imgsz=options['imgsz'], device=options['device'],
                           batch=4, project=str(run), name=name + '-validation', plots=True)
    report = {
        'variant': variant, 'checkpoint_sha256': digest(checkpoint),
        'initial_checkpoint_sha256': state['initial_checkpoint_sha256'],
        'split_manifest_sha256': split_hash, 'augmentation_sha256': augmentation_hash,
        'options': {**{k: v for k, v in options.items() if k not in ['data', 'project']},
                    'degrade_probability': state['degrade_probability']},
        'data_role': variant + '.yaml', 'training_metadata_sha256': digest(state_path),
        'resumed_from_checkpoint_sha256': resumed_hash,
        'validation': {'map50': float(metrics.box.map50), 'map50_95': float(metrics.box.map),
                       'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr), 'per_class': metrics.summary()},
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'ultralytics': ultralytics.__version__},
        'elapsed_seconds': time.monotonic() - start,
        'determinism': 'MPS includes nondeterministic operators; exact repeats are not guaranteed. External degradation RNG restarts from its recorded seed on resume.',
    }
    (run / (name + '-selection.json')).write_text(json.dumps(report, indent=2, default=lambda x: x.item()) + '\n')
    print(json.dumps(report['validation'], default=lambda x: x.item()), flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/performance-study')
    parser.add_argument('--grouped',type=Path,help='grouped-split directory written by prepare_grouped_split.py (default: artifacts/grouped)')
    parser.add_argument('--run',type=Path,default=ROOT/'runs/performance-study')
    parser.add_argument('--variant',choices=['base','improved'],default='improved')
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--device',default='mps',help='mps, cpu, or CUDA device index')
    parser.add_argument('--name',help='run name under --run (default: the variant); use a new name for each experiment. finalize_study.py only reads the canonical base/improved names, so name runs you mean to finalize accordingly')
    parser.add_argument('--init',type=Path,help='start from this checkpoint instead of the COCO yolov8s.pt (for example to fine-tune a finished run at a larger --imgsz)')
    parser.add_argument('--batch',type=int,default=8,help='training batch size; lower it for larger --imgsz')
    parser.add_argument('--hyp',type=json.loads,help='JSON object of extra Ultralytics train arguments, e.g. \'{"lr0":0.002,"cos_lr":true}\'')
    parser.add_argument('--degrade',type=probability,default=None,help='probability of random blur / noise / JPEG on each training image (0 disables)')
    parser.add_argument('--resume',action='store_true',help='continue the interrupted run called --name from its last.pt, with the settings stored in that checkpoint')
    args=parser.parse_args()
    if args.command=='prepare':
        m=prepare(args.output,args.grouped);n=augment(args.output);print(json.dumps({'counts':m['counts'],'final_groups':m['final_groups'],'ball_crops':n}))
    else:train(args.output,args.run,args.variant,args.epochs,args.imgsz,args.device,args.name,args.resume,args.init,args.batch,args.hyp,args.degrade)
