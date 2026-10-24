import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Src'))
from evaluate_fresh_test import CLASSES,collect_images,calculate_metrics


class FreshEvaluationTests(unittest.TestCase):
    def test_metrics_count_false_plastic_predictions(self):
        r=calculate_metrics(['plastic','metal','trash'],['plastic','plastic','trash'])
        self.assertEqual(r['correct'],2)
        self.assertAlmostEqual(r['accuracy'],2/3)
        self.assertEqual(r['per_class']['plastic']['precision'],.5)
        self.assertEqual(r['per_class']['plastic']['recall'],1.)
        self.assertAlmostEqual(r['per_class']['plastic']['f1'],2/3)

    def test_perfect_six_class_predictions(self):
        r=calculate_metrics(CLASSES,CLASSES)
        self.assertEqual(r['accuracy'],1.)
        self.assertEqual(r['macro_f1'],1.)

    def test_mismatched_predictions_fail(self):
        with self.assertRaises(ValueError): calculate_metrics(['glass'],[])

    def test_wrong_extraction_location_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'expected 5 images'):collect_images(temp)

    def test_identical_photos_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'cardboard';folder.mkdir()
            for i in range(5):Image.new('RGB',(48,48),'red').save(folder/f'{i}.png')
            with self.assertRaisesRegex(ValueError,'Duplicate evaluation'):collect_images(temp)


if __name__=='__main__':unittest.main()
