"""Prepare the Kaggle copy linked by the official D-Fire README; no re-splitting."""
import argparse
import hashlib
import json
import shutil
from PIL import Image, ImageOps
from pathlib import Path
import yaml
from ultralytics.data.utils import IMG_FORMATS

SOURCE = 'https://github.com/gaiasd/DFireDataset#computer-download-links'
HANDLE = 'sayedgamal99/smoke-fire-detection-yolo'


def writable_copy(root, destination):
    """Copy mounted data once, repair JPEG endings before hashing, and log changes."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if root == destination or root in destination.parents or destination.exists():
        raise ValueError('Use a new destination outside the source; existing copies are not overwritten')
    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < size + 1024 ** 3:
        raise OSError('Insufficient disk space for writable data copy plus 1 GiB reserve')
    shutil.copytree(root, destination)
    changes = []
    for path in sorted(destination.rglob('*')):
        if path.suffix.lower() not in ('.jpg', '.jpeg') or not path.is_file():
            continue
        with path.open('rb') as stream:
            stream.seek(-2, 2)
            ending = stream.read()
        if ending == b'\xff\xd9':
            continue
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        # Match pinned Ultralytics JPEG repair; fail visibly if decoding fails.
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            ImageOps.exif_transpose(image).save(path, 'JPEG', subsampling=0, quality=100)
        changes.append(dict(file=str(path.relative_to(destination)), before_sha256=before,
                            after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (destination / 'jpeg-repairs.json').write_text(json.dumps(changes, indent=2))
    print(f'Writable copy ready: {destination}; repaired JPEGs: {len(changes)}')
    return destination


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    source = root / 'data.yaml'
    if not source.is_file():
        raise FileNotFoundError(f'Downloaded package has no {source}; inspect its layout before training')
    config = yaml.safe_load(source.read_text())
    raw_names = config.get('names')
    if isinstance(raw_names, list):
        names = dict(enumerate(raw_names))
    elif isinstance(raw_names, dict):
        names = {int(k): v for k, v in raw_names.items()}
    else:
        raise ValueError('Source YAML must define class names; cannot guess class order')
    if set(names) != {0, 1} or {str(v).lower().strip() for v in names.values()} != {'fire', 'smoke'}:
        raise ValueError(f'Unexpected source classes {names}; inspect the package')
    splits, counts = {}, {}
    for split, alternatives in [('train', ('train',)), ('val', ('val', 'valid')), ('test', ('test',))]:
        candidates = [base / folder / 'images' for base in (root / 'data', root)
                      for folder in alternatives if (base / folder / 'images').is_dir()]
        if len(candidates) != 1:
            raise ValueError(f'Expected one {split}/images directory, found {candidates}')
        images = candidates[0]
        files = sorted(p for p in images.rglob('*') if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS)
        if not files or not (images.parent / 'labels').is_dir():
            raise ValueError(f'Empty images or missing labels directory: {images}')
        splits[split], counts[split] = str(images), len(files)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(path=str(root), **splits, names=names)
    output.write_text(yaml.safe_dump(result, sort_keys=False))
    provenance = dict(source=SOURCE, kaggle_handle=HANDLE, downloaded_root=str(root),
                      original_yaml_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      names=names, candidate_counts=counts,
                      protocol='Kaggle supplied split; NOT verified as HIFE authors exact split')
    output.with_suffix('.source.json').write_text(json.dumps(provenance, indent=2))
    print(yaml.safe_dump(result, sort_keys=False))
    print('Image candidates:', counts)
    print('Preserved downloaded splits. These are not verified HIFE paper manifests.')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.root, args.output)


if __name__ == '__main__':
    main()
