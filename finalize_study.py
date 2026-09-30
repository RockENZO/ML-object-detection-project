"""Freeze validation resolution, then run one final detector comparison."""
import argparse
import json
from pathlib import Path
from prepare_grouped_split import ROOT,NAMES,digest


def summarize(metrics):
    return {'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),'map50':float(metrics.box.map50),'map50_95':float(metrics.box.map),'per_class':metrics.summary()}


def validate_role(study,role):
    manifest=json.loads((study/'manifest.json').read_text())
    for record in manifest['records']:
        if record['split']!=role:continue
        for kind,key in [('images','image'),('labels','label')]:
            path=study/role/kind/Path(record[key]).name
            if not path.is_file() or digest(path)!=record[key+'_sha256']:
                raise ValueError('Split artifact changed: '+str(path))


def select(study,run,device="mps"):
    from ultralytics import YOLO
    output=run/'resolution-selection.json'
    if output.exists():raise ValueError('Resolution selection already frozen')
    training=json.loads((run/'improved-selection.json').read_text())
    checkpoint=run/'improved/weights/best.pt'
    if digest(checkpoint)!=training['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    if digest(study/'manifest.json')!=training['split_manifest_sha256']:raise ValueError('Training split changed')
    validate_role(study,'val')
    candidates=[]
    for size in [640,960,1280]:
        model=YOLO(str(checkpoint))
        results=model.val(data=str((study/'improved.yaml').resolve()),split='val',imgsz=size,batch=4,device=device,plots=True,project=str(run.resolve()),name='resolution-'+str(size))
        candidates.append({'imgsz':size,'validation':summarize(results)})
    chosen=max(candidates,key=lambda x:x['validation']['map50_95'])
    result={'criterion':'maximum validation mAP50-95 over fixed [640,960,1280] resolutions; no test selection',
            'candidates':candidates,'chosen':chosen,'checkpoint_sha256':digest(checkpoint),
            'split_manifest_sha256':digest(study/'manifest.json'),'final_test_evaluated':False}
    output.write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n')
    print(json.dumps(chosen,default=lambda x:x.item()),flush=True)


def final_test(study,run,device="mps"):
    from ultralytics import YOLO
    import yaml
    import yolo_inference
    output=run/'final-test-comparison.json'
    if output.exists():raise ValueError('Final comparison already evaluated')
    selection=json.loads((run/'resolution-selection.json').read_text());manifest=json.loads((study/'manifest.json').read_text())
    if digest(study/'manifest.json')!=selection['split_manifest_sha256']:raise ValueError('Split changed')
    if digest(run/'improved/weights/best.pt')!=selection['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    validate_role(study,'test')
    reports={}
    for variant,size in [('base',320),('improved',selection['chosen']['imgsz'])]:
        checkpoint=run/variant/'weights/best.pt';training=json.loads((run/(variant+'-selection.json')).read_text())
        if digest(checkpoint)!=training['checkpoint_sha256']:raise ValueError('Training checkpoint changed')
        destination=run/('final-'+variant)
        yolo_inference.main(['--model',str(checkpoint),'evaluate','--data',str(study/(variant+'.yaml')),'--output-dir',str(destination),'--imgsz',str(size),'--batch','4','--device',device])
        reports[variant]=json.loads((destination/'metrics.json').read_text())
    # Source-group breakdown of the selected detector, after all selection was frozen.
    by_group={}
    for group in manifest['final_groups']:
        records=[r for r in manifest['records'] if r['split']=='test' and r['group']==group]
        listing=study/('final-group-'+group+'.txt');listing.write_text(''.join(str((study/'test/images'/Path(r['image']).name).resolve())+'\n' for r in records))
        data=yaml.safe_load((study/'improved.yaml').read_text());data['test']=str(listing.resolve())
        config=study/('final-group-'+group+'.yaml');config.write_text(yaml.safe_dump(data))
        metrics=YOLO(str(run/'improved/weights/best.pt')).val(data=str(config.resolve()),split='test',imgsz=selection['chosen']['imgsz'],batch=4,device=device,plots=True,project=str(run.resolve()),name='final-group-'+group)
        by_group[group]={'images':len(records),'metrics':summarize(metrics)}
    result={'protocol':manifest['protocol'],'split_manifest_sha256':digest(study/'manifest.json'),
            'resolution_selection_sha256':digest(run/'resolution-selection.json'),'results':reports,'selected_by_group':by_group,
            'limitations':manifest['limitations']+['Only two provisional video-prefix groups in final test; these are not proven independent matches.',
               'Baseline and improved models were both newly trained on this study split; earlier models trained on final groups are excluded.']}
    output.write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n')
    print(json.dumps({variant:{k:result[k] for k in ['box_precision','box_recall','box_map50','box_map50_95']} for variant,result in reports.items()},indent=2),flush=True)

def export_reports(study,run):
    import csv
    import shutil
    import yaml
    from collections import Counter
    reports=ROOT/'reports'
    reports.mkdir(exist_ok=True)
    manifest=json.loads((study/'manifest.json').read_text())
    split={key:value for key,value in manifest.items() if key!='records'}
    split['manifest_sha256']=digest(study/'manifest.json')
    split['class_annotations']={role:dict(Counter(NAMES[c] for record in manifest['records'] if record['split']==role for c in record['classes'])) for role in ['train','val','test']}
    augmentation=json.loads((study/'augmentation.json').read_text())
    split['augmentation']={key:value for key,value in augmentation.items() if key!='crops'}
    split['augmentation']['crops']=len(augmentation['crops'])
    split['augmentation']['manifest_sha256']=digest(study/'augmentation.json')
    split['augmented_training_images']=manifest['counts']['train']+len(augmentation['crops'])
    training={}
    for variant in ['base','improved']:
        metadata=json.loads((run/(variant+'-selection.json')).read_text())
        curve=run/variant/'results.csv'
        rows=list(csv.DictReader(curve.open()))
        best=max(rows,key=lambda row:float(row['metrics/mAP50-95(B)']))
        resolved=yaml.safe_load((run/variant/'args.yaml').read_text())
        for key in ['data','model','project','save_dir']:
            if resolved.get(key):resolved[key]=Path(str(resolved[key])).name
        metadata['resolved_training_arguments']=resolved
        metadata['completed_epochs']=len(rows)
        metadata['best_validation_map50_95_epoch']=int(best['epoch'])
        metadata['curve_sha256']=digest(curve)
        training[variant]=metadata
        shutil.copy2(curve,reports/('study_curve_'+variant+'_20260930.csv'))
    for name,value in [('split',split),('training',training),('resolution',json.loads((run/'resolution-selection.json').read_text())),('test',json.loads((run/'final-test-comparison.json').read_text()))]:
        (reports/('study_'+name+'_20260930.json')).write_text(json.dumps(value,indent=2)+'\n')
    print('Exported frozen study evidence to reports/')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['select','test','report'])
    parser.add_argument('--device',default='mps')
    parser.add_argument('--study',type=Path,default=ROOT/'artifacts/performance-study')
    parser.add_argument('--run',type=Path,default=ROOT/'runs/performance-study')
    args=parser.parse_args()
    if args.command=='report':export_reports(args.study,args.run)
    else:(select if args.command=='select' else final_test)(args.study,args.run,args.device)
