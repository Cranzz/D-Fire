from ultralytics.models.yolo.detect import DetectionTrainer
from dfire.model import build_model


class ResearchTrainer(DetectionTrainer):
    variant = 'hife'

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.data['nc'] != 2:
            raise ValueError('D-Fire protocol requires exactly two verified classes')
        # Checkpoints retain variant metadata; resume cannot silently become baseline.
        variant = cfg.get('dfire_variant', self.variant) if isinstance(cfg, dict) else self.variant
        model = build_model(variant, self.data['nc'], self.args.seed, self.data['channels'])
        if weights is not None:
            model.load(weights)
        return model
