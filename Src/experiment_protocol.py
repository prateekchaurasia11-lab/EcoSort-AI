"""Compare exact experiment membership independently of machine-specific paths."""
from collections import Counter
import hashlib
import json


def membership(records):
    return Counter((r['source'],r['pixel_sha256'],r['label'],r['split']) for r in records)


def require_same_split(records,reference):
    actual,expected=membership(records),membership(reference)
    if actual!=expected:
        added=sum((actual-expected).values());missing=sum((expected-actual).values())
        raise ValueError(f'Dataset differs from the prior experiment ({added} added/changed, {missing} missing/changed). Training stopped to preserve the comparison.')
    canonical=sorted([list(key)+[count] for key,count in actual.items()])
    return hashlib.sha256(json.dumps(canonical,separators=(',',':')).encode()).hexdigest()
