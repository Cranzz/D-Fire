"""Freeze actual YOLO loader image paths, image and label hashes; refuse split overlap."""
import argparse
import hashlib
import json
from pathlib import Path
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.data.base import BaseDataset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = check_det_dataset(args.data, autodownload=False)
    records, seen = [], {}
    for split in ('train', 'val', 'test'):
        if not data.get(split):
            continue
        # Reuse pinned loader's path expansion (directories or txt lists).
        proxy = object.__new__(BaseDataset)
        proxy.prefix, proxy.fraction = '', 1.0
        images = proxy.get_img_files(data[split])
        for image, label in zip(images, img2label_paths(images)):
            digest = sha(image)
            if digest in seen and seen[digest] != split:
                raise ValueError(f'Image content overlap: {seen[digest]} / {split}: {image}')
            seen[digest] = split
            records.append(dict(split=split, image=str(Path(image).resolve()),
                                image_sha256=digest, label=str(Path(label).resolve()),
                                label_sha256=sha(label) if Path(label).exists() else None))
    result = dict(names=data['names'], images=records,
                  note='Raw loader candidates; inspect loader rejection logs separately.')
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'{len(records)} records; manifest SHA256 {sha(args.output)}')


if __name__ == '__main__':
    main()
