"""Validation-driven detector improvement on a newly reserved video-prefix holdout."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import time
from prepare_grouped_split import DATA, ROOT, NAMES, collect, digest


def prepare(output):
    if output.exists():
        raise ValueError('Study output already exists')
    previous=json.loads((ROOT/'artifacts/grouped/manifest.json').read_text())
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
              'seed':20260930,'previous_manifest_sha256':digest(ROOT/'artifacts/grouped/manifest.json'),
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


def train(output, run, variant, epochs, imgsz, device="mps"):
    import platform
    import torch
    import ultralytics
    from ultralytics import YOLO
    options=dict(data=str((output/(variant+'.yaml')).resolve()),epochs=epochs,imgsz=imgsz,batch=8,
                 seed=42,deterministic=True,device=device,workers=0,patience=20,
                 project=str(run.resolve()),name=variant,exist_ok=False,plots=True)
    start=time.monotonic()
    model=YOLO(str(ROOT/'models/yolov8s.pt'));model.train(**options)
    checkpoint=run/variant/'weights/best.pt'
    # validation only; final test evaluation is a separate explicit operation.
    selected=YOLO(str(checkpoint));metrics=selected.val(data=options['data'],split='val',imgsz=imgsz,device=device,batch=4,project=str(run.resolve()),name=variant+'-validation',plots=True)
    report={'variant':variant,'checkpoint_sha256':digest(checkpoint),'initial_checkpoint_sha256':digest(ROOT/'models/yolov8s.pt'),
            'split_manifest_sha256':digest(output/'manifest.json'),'augmentation_sha256':digest(output/'augmentation.json') if variant=='improved' else None,
            'options':{k:v for k,v in options.items() if k not in ['data','project']},'data_role':variant+'.yaml',
            'validation':{'map50':float(metrics.box.map50),'map50_95':float(metrics.box.map),'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),'per_class':metrics.summary()},
            'runtime':{'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__},'elapsed_seconds':time.monotonic()-start,
            'determinism':'MPS includes nondeterministic operators; exact repeats are not guaranteed.'}
    (run/(variant+'-selection.json')).write_text(json.dumps(report,indent=2,default=lambda x:x.item())+'\n')
    print(json.dumps(report['validation'],default=lambda x:x.item()),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/performance-study')
    parser.add_argument('--run',type=Path,default=ROOT/'runs/performance-study')
    parser.add_argument('--variant',choices=['base','improved'],default='improved')
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--device',default='mps',help='mps, cpu, or CUDA device index')
    args=parser.parse_args()
    if args.command=='prepare':
        m=prepare(args.output);n=augment(args.output);print(json.dumps({'counts':m['counts'],'final_groups':m['final_groups'],'ball_crops':n}))
    else:train(args.output,args.run,args.variant,args.epochs,args.imgsz,args.device)
