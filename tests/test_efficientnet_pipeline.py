import sys,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Src'))
from experiment_protocol import require_same_split
from test_taco_pipeline import image_bytes
from taco_data import CLASSES


class EfficientNetTests(unittest.TestCase):
    def test_split_lock_ignores_paths_and_order_but_rejects_reassignment(self):
        rows=[{'source':'realwaste','pixel_sha256':str(i),'label':'plastic','split':'train','file':'old/'+str(i)} for i in range(3)]
        copied=[dict(r,file='new/'+r['file']) for r in reversed(rows)]
        self.assertEqual(require_same_split(rows,rows),require_same_split(copied,rows))
        copied[0]['split']='test'
        with self.assertRaisesRegex(ValueError,'Dataset differs'): require_same_split(copied,rows)

    def test_split_lock_checks_multiplicity_and_labels(self):
        r={'source':'taco','pixel_sha256':'x','label':'plastic','split':'val'}
        with self.assertRaises(ValueError): require_same_split([r,r],[r])
        with self.assertRaises(ValueError): require_same_split([dict(r,label='paper')],[r])

    def test_two_stage_candidate_and_resnet_baseline_checkpoint_contract(self):
        import torch
        from efficientnet_model import build_model,load_checkpoint,inference_transform
        from train_efficientnet import train
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=[];seed=5000
            for split in ['train','val','test']:
                for source,label in [('trashnet',c) for c in CLASSES]+[('realwaste',c) for c in CLASSES]+[('taco','plastic')]*2:
                    p=root/f'{seed}.png';p.write_bytes(image_bytes(seed));seed+=1
                    rows.append({'file':str(p),'source':source,'label':label,'split':split,'source_category':'Other plastic wrapper'})
            baseline=build_model('resnet18');basefile=root/'baseline.pth'
            torch.save({'class_names':CLASSES,'model_state_dict':baseline.state_dict(),'image_size':224,
                        'mean':[.485,.456,.406],'std':[.229,.224,.225]},basefile)
            before=hashlib.file_digest(basefile.open('rb'),'sha256').hexdigest()
            captured={}
            def without_download(architecture,pretrained=False):
                model=build_model(architecture,pretrained=False)
                captured['frozen_before']=model.features[0].state_dict()
                captured['frozen_before']={k:v.clone() for k,v in captured['frozen_before'].items()}
                captured['classifier_before']=model.classifier[1].weight.detach().clone()
                return model
            with patch('train_efficientnet.build_model',side_effect=without_download):
                result=train(rows,root/'out',head_epochs=1,finetune_epochs=1,batch_size=14,device_name='cpu',
                             baseline_checkpoint=basefile,patience=2)
            model,ck=load_checkpoint(root/'out/ecosort_efficientnet_b0.pth')
            self.assertEqual(ck['architecture'],'efficientnet_b0')
            self.assertEqual(result['source_test']['realwaste']['n'],6)
            self.assertEqual(result['baseline_source_test']['realwaste']['n'],6)
            self.assertEqual(before,hashlib.file_digest(basefile.open('rb'),'sha256').hexdigest())
            for k,v in captured['frozen_before'].items(): self.assertTrue(torch.equal(v,model.features[0].state_dict()[k]))
            self.assertFalse(torch.equal(captured['classifier_before'],model.classifier[1].weight))
            from PIL import Image
            with torch.inference_mode(): scores=model(inference_transform(ck)(Image.new('RGB',(400,300))).unsqueeze(0))
            self.assertEqual(tuple(scores.shape),(1,6));self.assertTrue(torch.isfinite(scores).all())
            ck['class_names']=list(reversed(CLASSES));torch.save(ck,root/'bad.pth')
            with self.assertRaisesRegex(ValueError,'class order'):load_checkpoint(root/'bad.pth')


if __name__=='__main__':unittest.main()
