"""Fresh transfer-learning baseline; grouped holdout is never used for model selection."""
import argparse
import json
from pathlib import Path
import platform
import sys
from prepare_grouped_split import digest
import yolo_inference

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('artifacts/grouped/data.yaml'))
    parser.add_argument('--model',type=Path,default=Path('models/yolov8s.pt'))
    parser.add_argument('--output-dir',type=Path,default=Path('runs/grouped-baseline'))
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--imgsz',type=int,default=320)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Output directory already exists')
    manifest=args.data.parent/'manifest.json'
    if not manifest.is_file():parser.error('Generate a grouped split first')
    import ultralytics
    from ultralytics import YOLO
    import torch
    model=YOLO(str(args.model))
    options=dict(data=str(args.data.resolve()),epochs=args.epochs,imgsz=args.imgsz,batch=8,
                 seed=42,deterministic=True,device=args.device,workers=0,patience=10,
                 project=str(args.output_dir.resolve()),name='train',exist_ok=False)
    model.train(**options)
    checkpoint=args.output_dir/'train/weights/best.pt'
    metadata={'initial_checkpoint_sha256':digest(args.model),'trained_checkpoint_sha256':digest(checkpoint),
              'split_manifest_sha256':digest(manifest),'options':options,
              'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,
              'purpose':'Short transfer-learning baseline; not a converged production detector',
              'determinism_note':'Seed and split are fixed; MPS warns that some operations are nondeterministic. Exact metric reproduction is not guaranteed across hardware.'}
    # Save portable parameter values; absolute runtime paths are replaced by artifact roles.
    metadata['options']['data']='grouped/data.yaml';metadata['options']['project']='output-dir'
    (args.output_dir/'training_manifest.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    sys.exit(yolo_inference.main(['--model',str(checkpoint),'evaluate','--data',str(args.data),'--output-dir',str(args.output_dir/'test'),'--imgsz',str(args.imgsz)]))
