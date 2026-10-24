"""Add the six compatible RealWaste categories without changing existing splits."""
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from collections import Counter,defaultdict
from pathlib import Path,PurePosixPath
from PIL import Image,ImageOps,ImageDraw
from taco_data import CLASSES,UnionFind,dhash,pixel_hash

URL='https://archive.ics.uci.edu/static/public/908/realwaste.zip'
CITATION='Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G'
MAPPING={'Cardboard':'cardboard','Glass':'glass','Metal':'metal','Miscellaneous Trash':'trash','Paper':'paper','Plastic':'plastic'}


def download_realwaste(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and zipfile.is_zipfile(path): return path
    part=path.with_suffix('.zip.part')
    print('Downloading RealWaste (~657 MB).',flush=True)
    request=urllib.request.Request(URL,headers={'User-Agent':'EcoSort-AI research'})
    with urllib.request.urlopen(request,timeout=60) as src,part.open('wb') as dst:
        total=0;last=0
        while chunk:=src.read(1024*1024):
            dst.write(chunk);total+=len(chunk)
            if total-last>=64*1024*1024:
                print(f'RealWaste downloaded: {total//(1024*1024)} MiB',flush=True);last=total
    if not zipfile.is_zipfile(part): raise ValueError('RealWaste download is not a valid ZIP.')
    part.replace(path)
    return path


def split_groups(rows,uf,seed=2026):
    grouped=defaultdict(list)
    for i,r in enumerate(rows): grouped[uf.find(i)].append(r)
    strata=defaultdict(list)
    for group,members in grouped.items():
        # Cross-label near-duplicate groups remain intact; use a deterministic majority stratum.
        counts=Counter(r['label'] for r in members)
        dominant=sorted(counts,key=lambda c:(-counts[c],c))[0]
        strata[dominant].append(group)
    result={}
    for label in CLASSES:
        groups=sorted(strata[label]);random.Random(seed+CLASSES.index(label)).shuffle(groups)
        if len(groups)<7: raise ValueError(f'Only {len(groups)} source groups for {label}; inspect grouping before training.')
        holdout=max(1,round(.15*len(groups)))
        for i,group in enumerate(groups): result[group]='val' if i<holdout else 'test' if i<2*holdout else 'train'
    return result


def prepare_realwaste(archive,existing_records,output,block_size=20,near_threshold=4):
    """Group filename-adjacent blocks and near copies before generating any split."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cache=output/'source_cache';cache.mkdir(exist_ok=True)
    existing_hashes={r['pixel_sha256'] for r in existing_records}
    candidates=[];skipped=[]
    with zipfile.ZipFile(archive) as z:
        members=[n for n in z.namelist() if PurePosixPath(n).parent.name in MAPPING and PurePosixPath(n).suffix.lower() in ('.jpg','.jpeg','.png')]
        for i,name in enumerate(sorted(members)):
            label=MAPPING[PurePosixPath(name).parent.name]
            content=z.read(name)
            with Image.open(io.BytesIO(content)) as im:
                rgb=im.convert('RGB');digest=pixel_hash(rgb);fingerprint=dhash(rgb)
                if min(rgb.size)<96:
                    skipped.append({'archive_member':name,'reason':'image side below 96 pixels'});continue
            path=cache/f'{label}_{PurePosixPath(name).name}'
            path.write_bytes(content)
            number=re.search(r'(\d+)(?=\.[^.]+$)',name)
            if not number: raise ValueError(f'Cannot determine image sequence number: {name}')
            candidates.append({'file':str(path),'label':label,'source':'realwaste','archive_member':name,
                'pixel_sha256':digest,'dhash':fingerprint,'sequence_block':(int(number.group())-1)//block_size})
            if (i+1)%500==0: print(f'RealWaste inspected {i+1}/{len(members)} images',flush=True)
    uf=UnionFind(range(len(candidates)))
    blocks={};seen={};conflicting=set();duplicates=set();collisions=set()
    for i,r in enumerate(candidates):
        key=(r['label'],r['sequence_block'])
        if key in blocks: uf.union(i,blocks[key])
        else: blocks[key]=i
        digest=r['pixel_sha256']
        if digest in existing_hashes: collisions.add(i)
        if digest in seen:
            j=seen[digest];uf.union(i,j)
            if r['label']!=candidates[j]['label']: conflicting.update([i,j])
            else: duplicates.add(i)
        else: seen[digest]=i
        for j,previous in enumerate(candidates[:i]):
            if (r['dhash']^previous['dhash']).bit_count()<=near_threshold: uf.union(i,j)
    # A source-group collision with an existing experiment excludes the whole new group.
    blocked={uf.find(i) for i in collisions|conflicting}
    assignments=split_groups(candidates,uf)
    records=[]
    for i,r in enumerate(candidates):
        group=uf.find(i)
        reason=('group contains existing-dataset pixels or conflicting exact labels' if group in blocked else
                'exact duplicate within the same class' if i in duplicates else None)
        if reason: skipped.append(dict(r,reason=reason));continue
        records.append(dict(r,group_id=f'realwaste_{group}',split=assignments[group]))
    counts={s:dict(Counter(r['label'] for r in records if r['split']==s)) for s in ('train','val','test')}
    for s in counts:
        for c in CLASSES:
            if counts[s].get(c,0)<(30 if s=='train' else 10):
                raise ValueError(f'Insufficient RealWaste {s}/{c} after grouping/exclusion: {counts[s].get(c,0)}')
    digest=hashlib.file_digest(open(archive,'rb'),'sha256').hexdigest()
    manifest={'source_url':URL,'source_doi':'https://doi.org/10.24432/C5SS4G','citation':CITATION,
        'license':'CC BY 4.0','archive_sha256':digest,'class_mapping':MAPPING,
        'unmapped_classes_excluded':['Food Organics','Textile Trash','Vegetation'],
        'candidate_images':len(members),'usable_images':len(records),'counts':counts,
        'seed':2026,'sequence_block_size':block_size,'near_duplicate_dhash_threshold':near_threshold,
        'split_policy':'Approximately 70/15/15 by grouped source blocks, stratified on majority class. Groups combine within-class filename blocks and near-identical images.',
        'limitation':'Filename blocks are a conservative sequence heuristic, not verified capture sessions. Perceptual hashing cannot find every related scene. RealWaste comes from one landfill environment.',
        'records':records,'excluded':skipped}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    validate_splits(existing_records+records)
    preview(records,output/'training_preview.jpg')
    print('NEW RealWaste usable images:',len(records),flush=True)
    print('NEW RealWaste split totals:',dict(Counter(r['split'] for r in records)),flush=True)
    return manifest


def validate_splits(records):
    pixels=defaultdict(set);groups=defaultdict(set)
    for r in records:
        pixels[r['pixel_sha256']].add(r['split'])
        if 'group_id' in r: groups[(r['source'],r['group_id'])].add(r['split'])
    if any(len(v)>1 for v in pixels.values()): raise ValueError('Exact pixels cross splits.')
    if any(len(v)>1 for v in groups.values()): raise ValueError('Source groups cross splits.')


def preview(records,path):
    sheet=Image.new('RGB',(1000,840),'white');draw=ImageDraw.Draw(sheet)
    for ci,label in enumerate(CLASSES):
        pool=[r for r in records if r['split']=='train' and r['label']==label]
        for j,r in enumerate(random.Random(42).sample(pool,min(5,len(pool)))):
            with Image.open(r['file']) as im: image=ImageOps.contain(im.convert('RGB'),(190,110))
            x,y=j*200,ci*140;sheet.paste(image,(x,y));draw.text((x,y+115),label,fill='black')
    sheet.save(path)
