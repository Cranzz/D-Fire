import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from dfire.model import VARIANTS, build_model, parameter_report
from dfire.trainer import ResearchTrainer


def metadata():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(git_commit=commit, python=platform.python_version(), torch=torch.__version__,
                ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--model', choices=VARIANTS, default='hife')
    train = sub.add_parser('train')
    train.add_argument('--model', choices=VARIANTS, default='hife')
    train.add_argument('--data', required=True)
    train.add_argument('--config', default='configs/train.yaml')
    train.add_argument('--project', default='runs')
    train.add_argument('--name', required=True)
    train.add_argument('--seed', type=int, default=20260716)
    train.add_argument('--device', default='0')
    train.add_argument('--resume', type=Path)
    val = sub.add_parser('eval')
    val.add_argument('--weights', required=True)
    val.add_argument('--data', required=True)
    val.add_argument('--split', choices=['val', 'test'], default='val')
    val.add_argument('--device', default='0')
    val.add_argument('--output', default='outputs/evaluation')
    args = parser.parse_args()
    if args.command == 'audit':
        print(json.dumps(parameter_report(build_model(args.model)), indent=2))
    elif args.command == 'train':
        options = yaml.safe_load(Path(args.config).read_text())
        options.update(model='yolo11n.yaml', data=str(Path(args.data).resolve()),
                       project=args.project, name=args.name, seed=args.seed, device=args.device)
        if args.resume:
            options['resume'] = str(args.resume.resolve())
        trainer = ResearchTrainer(overrides=options)
        trainer.variant = args.model
        record = metadata()
        record.update(variant=(trainer.args.model if args.resume else args.model), requested_options=options,
                      resolved_options=vars(trainer.args),
                      config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                      data_yaml_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest())
        filename = 'resume-environment.json' if args.resume else 'environment.json'
        (trainer.save_dir / filename).write_text(json.dumps(record, indent=2, default=str))
        trainer.train()
    else:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        metrics = YOLO(args.weights).val(data=args.data, split=args.split, device=args.device,
                                         imgsz=640, batch=8, half=False, conf=.001,
                                         iou=.7, max_det=300, project=str(output), name='val')
        (output / 'metrics.json').write_text(json.dumps(dict(environment=metadata(),
                 weights_sha256=hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
                 split=args.split, metrics=metrics.results_dict), indent=2))


if __name__ == '__main__':
    main()
