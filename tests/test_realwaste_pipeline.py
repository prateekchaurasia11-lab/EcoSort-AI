import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Src'))
from taco_data import CLASSES,UnionFind
from realwaste_data import validate_splits,split_groups
from train_multisource import train
from test_taco_pipeline import image_bytes


class RealWasteTests(unittest.TestCase):
    def test_duplicate_pixels_crossing_sources_are_rejected(self):
        rows=[{'source':'trashnet','split':'train','pixel_sha256':'same'},
              {'source':'realwaste','split':'test','pixel_sha256':'same'}]
        with self.assertRaisesRegex(ValueError,'pixels cross'):validate_splits(rows)

    def test_grouped_splits_are_repeatable_and_preserve_whole_groups(self):
        rows=[]
        for c in CLASSES:
            for _ in range(12): rows.extend([{'label':c},{'label':c}])
        uf=UnionFind(range(len(rows)))
        for i in range(0,len(rows),2): uf.union(i,i+1)
        first=split_groups(rows,uf)
        self.assertEqual(first,split_groups(rows,uf))
        for label in CLASSES:
            groups={uf.find(i) for i,r in enumerate(rows) if r['label']==label}
            self.assertEqual({first[g] for g in groups},{'train','val','test'})

    def test_group_leakage_rejected_and_source_ids_namespaced(self):
        base={'pixel_sha256':'a','group_id':1,'source':'realwaste','split':'train'}
        validate_splits([base,dict(base,pixel_sha256='b',source='taco',split='test')])
        with self.assertRaisesRegex(ValueError,'groups cross'):
            validate_splits([base,dict(base,pixel_sha256='b',split='test')])

    def test_cpu_training_compares_baseline_on_same_new_source_and_saves_separately(self):
        import torch
        from torchvision import models
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=[];seed=3000
            for split in ['train','val','test']:
                entries=[('trashnet',c) for c in CLASSES]+[('realwaste',c) for c in CLASSES]+[('taco','plastic')]*2
                for source,label in entries:
                    p=root/f'{seed}.png';p.write_bytes(image_bytes(seed));seed+=1
                    rows.append({'file':str(p),'label':label,'split':split,'source':source,'source_category':'Other plastic wrapper'})
            original=models.resnet18
            baseline=original(weights=None);baseline.fc=torch.nn.Linear(512,6)
            checkpoint=root/'original.pth'
            torch.save({'model_state_dict':baseline.state_dict(),'class_names':CLASSES,
                        'mean':[.485,.456,.406],'std':[.229,.224,.225],'image_size':224},checkpoint)
            import hashlib
            before=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()
            with patch.object(models,'resnet18',side_effect=lambda **kw:original(weights=None)):
                result=train(rows,root/'new',head_epochs=1,finetune_epochs=1,batch_size=14,device_name='cpu',
                    checkpoint_name='new_model.pth',baseline_checkpoint=checkpoint,patience=1)
            self.assertEqual(result['source_test']['realwaste']['n'],6)
            self.assertEqual(result['baseline_source_test']['realwaste']['n'],6)
            self.assertEqual(result['baseline_source_test']['taco']['n'],2)
            self.assertEqual(set(result['source_test']),set(result['baseline_source_test']))
            self.assertTrue((root/'new/new_model.pth').exists())
            self.assertFalse((root/'new/ecosort_resnet18_taco_plastic.pth').exists())
            self.assertEqual(before,hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest())


if __name__=='__main__':unittest.main()
