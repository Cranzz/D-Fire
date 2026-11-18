from pathlib import Path
import yaml
import pytest
from dfire.prepare import prepare


def test_prepare_preserves_classes_and_split_and_ignores_foreign_paths(tmp_path):
    source = tmp_path / 'download'
    for split in ('train', 'valid', 'test'):
        images = source / 'data' / split / 'images'
        images.mkdir(parents=True)
        (images.parent / 'labels').mkdir()
        (images / 'a.jpg').write_bytes(b'fixture')
    (source / 'data.yaml').write_text(yaml.safe_dump(dict(names=['smoke', 'fire'],
                                                        train='/kaggle/foreign/train')))
    output = tmp_path / 'output/data.yaml'
    prepare(source, output)
    actual = yaml.safe_load(output.read_text())
    assert actual['names'] == {0: 'smoke', 1: 'fire'}
    assert actual['val'] == str(source / 'data/valid/images')
    assert Path(actual['train']).is_dir()
    assert output.with_suffix('.source.json').is_file()
    (source / 'data.yaml').write_text('names: [unknown, fire]')
    with pytest.raises(ValueError, match='classes'):
        prepare(source, output)
