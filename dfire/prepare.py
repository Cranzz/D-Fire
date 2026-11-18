"""Prepare the Kaggle copy linked by the official D-Fire README; no re-splitting."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml
from ultralytics.data.utils import IMG_FORMATS

SOURCE = 'https://github.com/gaiasd/DFireDataset#computer-download-links'
HANDLE = 'sayedgamal99/smoke-fire-detection-yolo'


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
