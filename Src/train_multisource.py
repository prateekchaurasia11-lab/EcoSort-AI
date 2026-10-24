"""Expanded-data ResNet18 training with early stopping and same-set baseline comparison."""
import csv
import json
import random
from collections import Counter
from pathlib import Path
from PIL import Image
from taco_data import CLASSES, POLICY


def train(records, output, head_epochs=3, finetune_epochs=7, batch_size=32, device_name="cuda",
          checkpoint_name="ecosort_resnet18_taco_plastic.pth", baseline_checkpoint=None, patience=None):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab before training. No training started.')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device(device_name)
    mean,std=[.485,.456,.406],[.229,.224,.225]
    evaluation=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean,std)])
    augmentation=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.75,1.)),transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),transforms.ColorJitter(.15,.15,.15,.05),transforms.ToTensor(),transforms.Normalize(mean,std)])
    class Images(Dataset):
        def __init__(self, rows, transform): self.rows,self.transform=rows,transform
        def __len__(self): return len(self.rows)
        def __getitem__(self,index):
            r=self.rows[index]
            with Image.open(r['file']) as im: image=self.transform(im.convert('RGB'))
            return image,CLASSES.index(r['label'])
    split_rows={s:[r for r in records if r['split']==s] for s in ('train','val','test')}
    generator=torch.Generator().manual_seed(42)
    loaders={s:DataLoader(Images(rows,augmentation if s=='train' else evaluation),batch_size=batch_size,
        shuffle=s=='train',num_workers=0,generator=generator if s=='train' else None) for s,rows in split_rows.items()}
    print('Device:',torch.cuda.get_device_name(0) if device_name=='cuda' else device_name,flush=True)
    print('Starting a NEW ImageNet-initialized ResNet18. No previous EcoSort weights loaded.',flush=True)
    model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc=nn.Linear(model.fc.in_features,len(CLASSES))
    model.to(device)
    counts=Counter(r['label'] for r in split_rows['train'])
    weights=torch.tensor([len(split_rows['train'])/(len(CLASSES)*counts[c]) for c in CLASSES],device=device)
    loss_fn=nn.CrossEntropyLoss(weight=weights)
    def predict(rows):
        model.eval()
        truth,prediction=[],[]
        with torch.inference_mode():
            for x,y in DataLoader(Images(rows,evaluation),batch_size=batch_size,num_workers=0):
                p=model(x.to(device)).argmax(1).cpu().tolist()
                prediction.extend(p)
                truth.extend(y.tolist())
        return truth,prediction
    def metrics(rows):
        y,p=predict(rows)
        result={'n':len(y),'accuracy':accuracy_score(y,p),
            'confusion_matrix':confusion_matrix(y,p,labels=list(range(6))).tolist()}
        if {r['label'] for r in rows}=={'plastic'}:
            result['plastic_recall']=result['accuracy']
            result['note']='Positive-only wrapper set: this measures recall, not six-class accuracy or plastic precision.'
            result['by_source_category']={c:{'n':sum(r.get('source_category')==c for r in rows),
                'plastic_recall':sum(pp==4 for r,pp in zip(rows,p) if r.get('source_category')==c)/sum(r.get('source_category')==c for r in rows)}
                for c in sorted({r['source_category'] for r in rows})}
        else:
            result['macro_f1']=f1_score(y,p,labels=list(range(6)),average='macro',zero_division=0)
            result['class_report']=classification_report(y,p,labels=list(range(6)),target_names=CLASSES,output_dict=True,zero_division=0)
        return result
    history=[]
    best=(-1.,-1.)
    checkpoint=output/checkpoint_name
    best_epoch=0
    epoch=0
    for stage,n_epochs in [('head',head_epochs),('layer4',finetune_epochs)]:
        if stage=='layer4':
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['model_state_dict'])
        for name,param in model.named_parameters():
            param.requires_grad=name.startswith('fc.') or (stage=='layer4' and name.startswith('layer4.'))
        groups=[{'params':model.fc.parameters(),'lr':1e-3 if stage=='head' else 1e-4}]
        if stage=='layer4': groups.append({'params':model.layer4.parameters(),'lr':1e-5})
        optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
        for _ in range(n_epochs):
            epoch+=1
            # Frozen layers and their batch-normalization statistics remain frozen.
            model.eval()
            model.fc.train()
            if stage=='layer4': model.layer4.train()
            total_loss,correct,n=0.,0,0
            for x,y in loaders['train']:
                x,y=x.to(device),y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model(x)
                loss=loss_fn(logits,y)
                loss.backward()
                optimizer.step()
                total_loss+=loss.item()*len(y)
                correct+=(logits.argmax(1)==y).sum().item()
                n+=len(y)
            val=metrics(split_rows['val'])
            wrapper_val=metrics([r for r in split_rows['val'] if r['source']=='taco'])
            row={'epoch':epoch,'stage':stage,'train_loss':total_loss/n,'train_accuracy':correct/n,
                'val_accuracy':val['accuracy'],'val_macro_f1':val['macro_f1'],'wrapper_val_recall':wrapper_val['plastic_recall']}
            history.append(row)
            print(f"Epoch {epoch}/{head_epochs+finetune_epochs}: val F1 {val['macro_f1']:.3f}, wrapper val recall {wrapper_val['plastic_recall']:.1%}",flush=True)
            score=(val['macro_f1'],val['accuracy'])
            if score>best:
                best=score
                best_epoch=epoch
                torch.save({'model_state_dict':model.state_dict(),'classes':CLASSES,'class_names':CLASSES,
                    'epoch':epoch,'val_macro_f1':best[0],'image_size':224,'mean':mean,'std':std,
                    'label_policy':POLICY,'initialization':'ImageNet1K V1; no previous EcoSort checkpoint'},checkpoint)
            if stage=='layer4' and patience is not None and epoch-best_epoch>=patience:
                print('Early stopping: validation selection score has not improved.',flush=True)
                break
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['model_state_dict'])
    source_rows={s:[r for r in split_rows['test'] if r['source']==s] for s in sorted({r['source'] for r in split_rows['test']})}
    source_test={s:metrics(rows) for s,rows in source_rows.items()}
    baseline_test=None
    if baseline_checkpoint is not None:
        base=torch.load(baseline_checkpoint,map_location=device,weights_only=True)
        if base['class_names']!=CLASSES or base.get('mean')!=mean or base.get('std')!=std or base.get('image_size')!=224:
            raise ValueError('Baseline class order or preprocessing differs; comparison stopped.')
        model.load_state_dict(base['model_state_dict'])
        baseline_test={s:metrics(rows) for s,rows in source_rows.items()}
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['model_state_dict'])
    final={'classes':CLASSES,'label_policy':POLICY,'selection':'highest combined validation macro F1, accuracy breaks ties',
        'selected_epoch':torch.load(checkpoint,map_location='cpu',weights_only=True)['epoch'],
        'trashnet_test':source_test['trashnet'],
        'new_wrapper_test':source_test['taco'],
        'source_test':source_test,'baseline_source_test':baseline_test,
        'combined_test':metrics(split_rows['test']),
        'limitations':['TrashNet test has been inspected in earlier experiments; interpret as diagnostic.',
            'TACO holdout is grouped by source and near-identical scenes, not independently collected deployment photos.',
            'Packaging categories are project labels, not verified material composition or recycling advice.',
            'TACO crops differ from whole-image TrashNet photos; evaluate fresh user photos before deployment.']}
    (output/'metrics.json').write_text(json.dumps(final,indent=2))
    with (output/'training_history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    print('Selected epoch:',final['selected_epoch'],flush=True)
    print('TrashNet test accuracy:',f"{final['trashnet_test']['accuracy']:.1%}",flush=True)
    print('NEW wrapper test recall:',f"{final['new_wrapper_test']['plastic_recall']:.1%}",flush=True)
    if 'realwaste' in source_test:
        print('NEW RealWaste test accuracy:',f"{source_test['realwaste']['accuracy']:.1%}",flush=True)
        if baseline_test is not None:
            print('Previous model on SAME RealWaste test:',f"{baseline_test['realwaste']['accuracy']:.1%}",flush=True)
    return final
