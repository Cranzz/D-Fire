"""Tiny synthetic integration test. Its metrics have no research meaning."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from ultralytics import YOLO
from dfire.trainer import ResearchTrainer
from dfire.cli import main


def test_training_interruption_resume_and_eval(tmp_path, monkeypatch):
    for split in ('train', 'val'):
        images = tmp_path / 'images' / split
        labels = tmp_path / 'labels' / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(2):
            Image.fromarray(np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(images / f'{i}.jpg')
            (labels / f'{i}.txt').write_text(f'{i} 0.5 0.5 0.3 0.3\n')
    data = tmp_path / 'data.yaml'
    data.write_text(yaml.safe_dump(dict(path=str(tmp_path), train='images/train', val='images/val',
                                       names={0: 'synthetic_a', 1: 'synthetic_b'})))
    options = dict(model='yolo11n.yaml', data=str(data), project=str(tmp_path / 'runs'),
                   name='smoke', epochs=2, batch=2, imgsz=64, device='cpu', workers=0,
                   amp=False, pretrained=False, plots=False, save=True,
                   optimizer='AdamW', lr0=.001, deterministic=True, seed=20260716,
                   mosaic=0., close_mosaic=0, val=True)
    trainer = ResearchTrainer(overrides=options)
    trainer.variant = 'hife'

    class SimulatedInterruption(Exception):
        pass

    def stop_after_checkpoint(t):
        if t.epoch == 0:
            raise SimulatedInterruption()

    trainer.add_callback('on_model_save', stop_after_checkpoint)
    try:
        trainer.train()
    except SimulatedInterruption:
        pass
    else:
        raise AssertionError('Interruption callback did not run')
    last = trainer.save_dir / 'weights/last.pt'
    assert last.is_file()
    resumed = ResearchTrainer(overrides=dict(resume=str(last), device='cpu'))
    resumed.train()
    assert resumed.start_epoch == 1
    assert resumed.model.yaml['dfire_variant'] == 'hife'
    model = YOLO(resumed.best)
    metrics = model.val(data=str(data), device='cpu', imgsz=64, batch=2, workers=0, plots=False,
                        project=str(tmp_path / 'eval'), name='smoke')
    assert 'metrics/mAP50(B)' in metrics.results_dict
