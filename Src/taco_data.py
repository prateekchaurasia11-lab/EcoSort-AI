"""TACO plastic-packaging crops, with source-grouped splits and provenance."""
import concurrent.futures
import hashlib
import io
import json
import math
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

TACO_COMMIT = '70479ad9adc3da971a8ae75ac97eb1026910eec7'
ANNOTATIONS_URL = f'https://raw.githubusercontent.com/pedropro/TACO/{TACO_COMMIT}/data/annotations.json'
CATEGORIES = {'Plastic film', 'Other plastic wrapper', 'Crisp packet'}
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
POLICY = ('EcoSort plastic includes visually identified flexible plastic packaging, including '
          'metallized snack packets, following the project packaging convention. '
          'Source labels do not verify chemical composition or recyclability.')

def sha256(data):
    return hashlib.sha256(data).hexdigest()

def pixel_hash(image):
    rgb = image.convert('RGB')
    return sha256(str(rgb.size).encode() + rgb.tobytes())

def dhash(image):
    small = image.convert('L').resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | (pixels[y*9+x] > pixels[y*9+x+1])
    return result

def crop_box(bbox, original_size, downloaded_size):
    x, y, width, height = bbox
    sx, sy = downloaded_size[0]/original_size[0], downloaded_size[1]/original_size[1]
    return (max(0, math.floor(x*sx)), max(0, math.floor(y*sy)),
            min(downloaded_size[0], math.ceil((x+width)*sx)),
            min(downloaded_size[1], math.ceil((y+height)*sy)))

def fetch_bytes(url, timeout=12):
    request = urllib.request.Request(url, headers={'User-Agent': 'EcoSort-AI research dataset preparation'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()

class UnionFind:
    def __init__(self, values):
        self.parent = {v: v for v in values}
    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

def grouped_splits(groups, seed=42):
    """Input maps group IDs to crisp-packet presence; every group stays in one split."""
    assignments = {}
    for has_crisp in (False, True):
        ids = sorted(k for k, flag in groups.items() if flag == has_crisp)
        random.Random(seed + int(has_crisp)).shuffle(ids)
        n = len(ids)
        n_val = max(1, round(n*.15)) if n >= 7 else 0
        n_test = n_val
        for index, group_id in enumerate(ids):
            assignments[group_id] = ('val' if index < n_val else
                                      'test' if index < n_val+n_test else 'train')
    return assignments

def prepare_taco(output, annotation_bytes=None, workers=12, min_side=48):
    """Download official preview images; keep only resolvable crops >= min_side pixels."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = annotation_bytes if annotation_bytes is not None else fetch_bytes(ANNOTATIONS_URL)
    annotations = json.loads(raw)
    (output/'annotations_source.json').write_bytes(raw)
    categories = {c['id']: c['name'] for c in annotations['categories']}
    images = {im['id']: im for im in annotations['images']}
    selected = [a for a in annotations['annotations'] if categories[a['category_id']] in CATEGORIES]
    by_image = defaultdict(list)
    for ann in selected:
        by_image[ann['image_id']].append(ann)
    cache = output/'source_cache'
    cache.mkdir(exist_ok=True)
    def download(image_id):
        info = images[image_id]
        path = cache/f'image_{image_id:05d}.jpg'
        urls = list(dict.fromkeys(u for u in (info.get('flickr_640_url'), info.get('flickr_url')) if u))
        if path.exists():
            with Image.open(path) as im:
                image = im.convert('RGB')
            if max(image.size) > 640:
                image = ImageOps.contain(image, (640, 640), Image.Resampling.LANCZOS)
                image.save(path, quality=95)
                with Image.open(path) as im:
                    image = im.convert('RGB')
            return image_id, image, (path.with_suffix('.url').read_text() if path.with_suffix('.url').exists() else 'cached URL unknown'), None
        failures = []
        for url in urls:
            for attempt in range(1):
                try:
                    content = fetch_bytes(url)
                    with Image.open(io.BytesIO(content)) as im:
                        image = ImageOps.contain(im.convert('RGB'), (640, 640), Image.Resampling.LANCZOS)
                    image.save(path, quality=95)
                    path.with_suffix('.url').write_text(url)
                    # Return the same cached decoded pixels on initial and repeat runs.
                    with Image.open(path) as im:
                        return image_id, im.convert('RGB'), url, None
                except Exception as exc:
                    failures.append(type(exc).__name__ + ': ' + str(exc))
        return image_id, None, None, failures
    downloaded = {}
    errors = []
    ids = sorted(by_image)
    print(f'TACO: {len(selected)} annotated candidate objects in {len(ids)} source images.', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (image_id, image, url, error) in enumerate(pool.map(download, ids), 1):
            if image is None:
                errors.append({'image_id': image_id, 'errors': error})
            else:
                downloaded[image_id] = (image, url)
            if i % 25 == 0 or i == len(ids):
                print(f'Downloaded/checking {i}/{len(ids)} sources; failures {len(errors)}.', flush=True)
    # Group exact copies and near-identical whole scenes BEFORE deciding splits.
    uf = UnionFind(sorted(downloaded))
    fingerprints = {}
    for image_id in sorted(downloaded):
        image, _ = downloaded[image_id]
        fingerprints[image_id] = (pixel_hash(image), dhash(image))
    scene_ids = sorted(downloaded)
    for i, a in enumerate(scene_ids):
        for b in scene_ids[:i]:
            if fingerprints[a][0] == fingerprints[b][0] or (fingerprints[a][1]^fingerprints[b][1]).bit_count() <= 4:
                uf.union(a, b)
    candidates = []
    crop_fingerprints = {}
    skipped = Counter()
    for image_id in scene_ids:
        image, url = downloaded[image_id]
        info = images[image_id]
        for ann in sorted(by_image[image_id], key=lambda x:x['id']):
            box = crop_box(ann['bbox'], (info['width'], info['height']), image.size)
            if box[2]-box[0] < min_side or box[3]-box[1] < min_side:
                skipped['too_small'] += 1
                continue
            crop = image.crop(box)
            digest = pixel_hash(crop)
            if digest in crop_fingerprints:
                uf.union(image_id, crop_fingerprints[digest])
                skipped['exact_crop_duplicate'] += 1
                continue
            crop_fingerprints[digest] = image_id
            candidates.append((image_id, ann, crop, digest, url))
    group_flags = {}
    for image_id, ann, crop, digest, url in candidates:
        gid = uf.find(image_id)
        group_flags[gid] = group_flags.get(gid, False) or categories[ann['category_id']] == 'Crisp packet'
    split_map = grouped_splits(group_flags)
    records = []
    for image_id, ann, crop, digest, url in candidates:
        split = split_map[uf.find(image_id)]
        relative = f'{split}/plastic/taco_{image_id:05d}_{ann["id"]:05d}.png'
        path = output/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(path)
        records.append({'file':relative,'image_id':image_id,'annotation_id':ann['id'],
            'group_id':uf.find(image_id),'split':split,'label':'plastic',
            'source_category':categories[ann['category_id']], 'bbox':ann['bbox'],
            'downloaded_crop_size':list(crop.size),'pixel_sha256':digest,
            'download_url':url,'original_url':images[image_id].get('flickr_url'),
            'license_id':images[image_id].get('license')})
    counts = Counter(r['split'] for r in records)
    (output/'download_errors.json').write_text(json.dumps(errors,indent=2))
    for split in ('train','val','test'):
        if counts[split] < 15:
            raise RuntimeError(f'Too few usable TACO {split} crops ({counts[split]}). No training started. Review download_errors.json and source availability.')
    manifest = {'taco_commit':TACO_COMMIT,'annotations_url':ANNOTATIONS_URL,
        'annotations_sha256':sha256(raw),'candidate_objects':len(selected),
        'downloaded_source_images':len(downloaded),'failed_source_images':len(errors),
        'counts':dict(counts),'source_category_counts':dict(Counter(r['source_category'] for r in records)),
        'skip_counts':dict(skipped),'min_crop_side':min_side,'seed':42,
        'split_unit':'source scene / exact crop duplicate / whole-image dHash distance <= 4',
        'label_policy':POLICY, 'licenses':annotations.get('licenses',[]),'records':records}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'download_errors.json').write_text(json.dumps(errors,indent=2))
    print('Usable new TACO crops:', dict(counts), 'Total:', len(records), flush=True)
    # Show only training examples; leave test images and scores untouched until final evaluation.
    sample=random.Random(42).sample([r for r in records if r['split']=='train'],min(30,counts['train']))
    sheet=Image.new('RGB',(900,6*145),'white')
    draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(sample):
        with Image.open(output/r['file']) as im:
            preview=ImageOps.contain(im,(170,115))
        x,y=(i%5)*180,(i//5)*145
        sheet.paste(preview,(x,y))
        draw.text((x,y+117),f'{r["annotation_id"]} {r["source_category"][:19]}',fill='black')
    sheet.save(output/'training_preview.jpg')
    return manifest

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    prepare_taco(args.output)
