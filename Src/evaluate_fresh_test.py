"""Compare saved EcoSort models on the six fresh_test folders; never train."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

CLASSES=['cardboard','glass','metal','paper','plastic','trash']
SUFFIXES={'.jpg','.jpeg','.png','.webp'}


def collect_images(root):
    from PIL import Image
    records=[];seen={}
    for label in CLASSES:
        folder=Path(root)/label
        paths=sorted(p for p in folder.glob('*') if p.is_file() and p.suffix.lower() in SUFFIXES)
        if len(paths)!=5:
            raise ValueError(f'{folder}: expected 5 images, found {len(paths)}. Check the extraction location.')
        for path in paths:
            with Image.open(path) as image:
                rgb=image.convert('RGB')
                digest=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            if digest in seen: raise ValueError(f'Duplicate evaluation photos: {seen[digest]} and {path}')
            seen[digest]=str(path)
            records.append({'path':path,'expected':label,'pixel_sha256':digest})
    return records


def calculate_metrics(truth,predictions):
    if not truth or len(truth)!=len(predictions): raise ValueError('Expected nonempty matching label lists.')
    matrix=[[0]*len(CLASSES) for _ in CLASSES]
    for expected,predicted in zip(truth,predictions): matrix[CLASSES.index(expected)][CLASSES.index(predicted)]+=1
    per_class={}
    for i,label in enumerate(CLASSES):
        correct=matrix[i][i];support=sum(matrix[i]);predicted=sum(row[i] for row in matrix)
        precision=correct/predicted if predicted else 0.
        recall=correct/support if support else 0.
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        per_class[label]={'correct':correct,'support':support,'precision':precision,'recall':recall,'f1':f1}
    correct=sum(matrix[i][i] for i in range(len(CLASSES)))
    return {'correct':correct,'n':len(truth),'accuracy':correct/len(truth),
            'macro_f1':sum(r['f1'] for r in per_class.values())/len(CLASSES),
            'classes':CLASSES,'confusion_matrix':matrix,'per_class':per_class}


def load_model(path,architecture):
    import torch
    from torchvision import models
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('class_names')!=CLASSES: raise ValueError(f'{path}: unexpected class order.')
    if checkpoint.get('image_size')!=224 or checkpoint.get('mean')!=[.485,.456,.406] or checkpoint.get('std')!=[.229,.224,.225]:
        raise ValueError(f'{path}: unexpected preprocessing metadata.')
    if checkpoint.get('architecture',architecture)!=architecture: raise ValueError(f'{path}: wrong architecture.')
    if architecture=='resnet18':
        model=models.resnet18(weights=None);model.fc=torch.nn.Linear(model.fc.in_features,len(CLASSES))
    elif architecture=='efficientnet_b0':
        model=models.efficientnet_b0(weights=None)
        model.classifier[1]=torch.nn.Linear(model.classifier[1].in_features,len(CLASSES))
    else: raise ValueError('Unsupported architecture.')
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    return model.eval()


def run(project):
    import torch
    from torchvision import transforms
    from PIL import Image
    project=Path(project).resolve()
    root=project/'data/fresh_test'
    paths={'resnet18':project/'evaluation_models/resnet18/Models/ecosort_resnet18_realwaste.pth',
           'efficientnet_b0':project/'evaluation_models/efficientnet/Models/ecosort_efficientnet_b0.pth'}
    for path in paths.values():
        if not path.is_file():raise FileNotFoundError(f'Model missing: {path}. Extract its result ZIP into evaluation_models first.')
    records=collect_images(root)
    torch.set_num_threads(min(4,max(1,os.cpu_count() or 1)))
    print('Found 30 photos, five per class. Running saved models on CPU; no training.',flush=True)
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                  transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    rows=[{'image':str(r['path'].relative_to(root)),'expected':r['expected'],'pixel_sha256':r['pixel_sha256']} for r in records]
    report={'dataset':'data/fresh_test','candidate_folder_labels':True,
            'label_counts':dict(Counter(r['expected'] for r in records)),'models':{},
            'limits':['Small 30-image web sample, not a representative final benchmark.',
                      'All trash examples are cigarette butts; some images show clean packaging.',
                      'Folder labels are project candidates, not chemically verified material composition.',
                      'Model scores are not calibrated probabilities of correctness. Keep these photos out of training.']}
    truth=[r['expected'] for r in records]
    for architecture,path in paths.items():
        print(f'Loading {architecture}...',flush=True)
        model=load_model(path,architecture);predictions=[]
        with torch.inference_mode():
            for index,record in enumerate(records):
                with Image.open(record['path']) as image:x=transform(image.convert('RGB')).unsqueeze(0)
                scores=model(x).softmax(1)[0]
                if not torch.isfinite(scores).all():raise ValueError('Non-finite model scores; evaluation stopped.')
                top=scores.argmax().item();predicted=CLASSES[top];predictions.append(predicted)
                rows[index][architecture+'_prediction']=predicted
                rows[index][architecture+'_correct']=predicted==record['expected']
                rows[index][architecture+'_score']=float(scores[top])
                for i,label in enumerate(CLASSES):rows[index][architecture+'_'+label+'_score']=float(scores[i])
        metrics=calculate_metrics(truth,predictions)
        with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        report['models'][architecture]=dict(metrics,checkpoint=str(path),checkpoint_sha256=digest)
        print(f"{architecture}: {metrics['correct']}/30 correct ({metrics['accuracy']:.2%}); macro F1 {metrics['macro_f1']:.4f}",flush=True)
        for label,value in metrics['per_class'].items():print(f"  {label}: {value['correct']}/{value['support']}",flush=True)
        del model
    report['model_disagreement_count']=sum(r['resnet18_prediction']!=r['efficientnet_b0_prediction'] for r in rows)
    report['evaluation_image_fingerprints']=[{'image':r['image'],'expected':r['expected'],'pixel_sha256':r['pixel_sha256']} for r in rows]
    output=project/'Results'/('fresh_test_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (output/'predictions.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Model disagreements:',report['model_disagreement_count'],flush=True)
    print('Results saved to:',output,flush=True)
    print('Upload comparison.json here for review. No model was modified.',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path.cwd(),help='EcoSort-AI project folder; defaults to the current folder.')
    args=parser.parse_args()
    try:run(args.project)
    except (ValueError,FileNotFoundError) as error:parser.exit(1,f'Error: {error}\n')
