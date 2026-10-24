"""Data leakage checks and an actual synthetic CPU optimizer/checkpoint smoke test."""
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Src'))
from taco_data import CLASSES, crop_box, grouped_splits, pixel_hash
from train_taco import read_trashnet, prepare_records, train


def image_bytes(seed):
    import random
    rng=random.Random(seed)
    im=Image.frombytes('RGB',(64,64),rng.randbytes(64*64*3))
    f=io.BytesIO();im.save(f,format='PNG');return f.getvalue()


class DataTests(unittest.TestCase):
    def test_crop_coordinates_use_downloaded_resolution(self):
        self.assertEqual(crop_box([100,50,300,200],(1000,500),(640,320)),(64,32,256,160))
        self.assertEqual(crop_box([-20,-10,1100,600],(1000,500),(640,320)),(0,0,640,320))

    def test_group_split_is_deterministic_and_keeps_holdouts(self):
        groups={i:i%3==0 for i in range(120)}
        first=grouped_splits(groups)
        self.assertEqual(first,grouped_splits(dict(reversed(list(groups.items())))))
        self.assertEqual(set(first),set(groups))
        for split in ['train','val','test']:
            self.assertTrue(any(first[i]==split and groups[i] for i in groups))

    def test_all_conflicting_copies_removed_and_archive_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'input.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for split_i,split in enumerate(['train','val','test']):
                    for class_i,label in enumerate(CLASSES):
                        z.writestr(f'split/{split}/{label}/ok.png',image_bytes(split_i*6+class_i))
                duplicate=image_bytes(100)
                z.writestr('split/train/glass/duplicate.png',duplicate)
                z.writestr('split/val/plastic/duplicate.png',duplicate)
            original=archive.read_bytes()
            records,excluded=read_trashnet(archive,root/'extracted')
            self.assertEqual(len(records),18)
            self.assertEqual(len(excluded),2)
            self.assertEqual(original,archive.read_bytes())

    def test_bad_zip_cannot_write_outside_selected_class_folders(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z: z.writestr('../../escape.py','x')
            with self.assertRaises(ValueError): read_trashnet(archive,root/'extracted')
            self.assertFalse((root/'escape.py').exists())

    def test_training_rejects_tiny_wrapper_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'input.zip';taco=root/'taco';taco.mkdir()
            with zipfile.ZipFile(archive,'w') as z:
                for si,s in enumerate(['train','val','test']):
                    for ci,c in enumerate(CLASSES): z.writestr(f'{s}/{c}/ok.png',image_bytes(si*6+ci))
            records=[{'group_id':i,'file':f'{i}.png','split':s,'label':'plastic','pixel_sha256':str(i)} for i,s in enumerate(['train','val','test'])]
            (taco/'manifest.json').write_text(json.dumps({'records':records}))
            with self.assertRaisesRegex(ValueError,'Not enough NEW'): prepare_records(archive,taco,root/'work')


class TrainingSmokeTest(unittest.TestCase):
    def test_two_stages_selection_save_reload_and_final_metrics_on_cpu(self):
        import torch
        from torchvision import models
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);records=[]
            seed=1000
            for split in ['train','val','test']:
                for label in CLASSES+['plastic','plastic']:
                    path=root/f'{seed}.png';path.write_bytes(image_bytes(seed));seed+=1
                    records.append({'file':str(path),'label':label,'split':split,
                        'source':'taco' if label=='plastic' and len([r for r in records if r['split']==split and r['label']=='plastic']) else 'trashnet',
                        'source_category':'Other plastic wrapper'})
            real_resnet=models.resnet18
            # Avoid downloading ImageNet weights for this synthetic functional test.
            with patch.object(models,'resnet18',side_effect=lambda **kwargs:real_resnet(weights=None)):
                result=train(records,root/'result',head_epochs=1,finetune_epochs=1,batch_size=8,device_name='cpu')
            self.assertIn(result['selected_epoch'],[1,2])
            self.assertEqual(result['new_wrapper_test']['n'],2)
            checkpoint=torch.load(root/'result/ecosort_resnet18_taco_plastic.pth',weights_only=True)
            self.assertEqual(checkpoint['class_names'],CLASSES)
            model=real_resnet(weights=None);model.fc=torch.nn.Linear(512,6)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.assertTrue((root/'result/metrics.json').is_file())
            self.assertEqual(len((root/'result/training_history.csv').read_text().splitlines()),3)


if __name__=='__main__': unittest.main()
