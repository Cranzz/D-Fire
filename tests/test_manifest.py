import json
import sys
import numpy as np
import pytest
import yaml
from PIL import Image
from dfire.manifest import main


def test_manifest_and_overlap(tmp_path, monkeypatch):
    for split in ('train', 'val'):
        images = tmp_path / 'images' / split
        images.mkdir(parents=True)
        value = 0 if split == 'train' else 255
        Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(images / 'a.png')
    data = tmp_path / 'data.yaml'
    data.write_text(yaml.safe_dump(dict(path=str(tmp_path), train='images/train', val='images/val',
                                       names={0: 'a', 1: 'b'})))
    output = tmp_path / 'manifest.json'
    monkeypatch.setattr(sys, 'argv', ['manifest', '--data', str(data), '--output', str(output)])
    main()
    records = json.loads(output.read_text())['images']
    assert len(records) == 2 and records[0]['label_sha256'] is None
    (tmp_path / 'images/val/a.png').write_bytes((tmp_path / 'images/train/a.png').read_bytes())
    with pytest.raises(ValueError, match='overlap'):
        main()
