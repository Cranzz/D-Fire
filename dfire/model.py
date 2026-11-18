"""Build the pinned upstream graph, then replace typed modules; no parser patching."""
from copy import deepcopy
import torch
from torch import nn
from ultralytics.nn.modules import C3k2, Conv, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import init_seeds
from dfire.modules import MSGhostC3k2, RetentionGSConv, CTD

VARIANTS = ('baseline', 'msghost', 'retention', 'ctd', 'hife')


class CTDDetect(Detect):
    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return {}
        adapted = [adapter(feature) for adapter, feature in zip(self.adapters, x)]
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](pair[0]).view(bs, 4 * self.reg_max, -1)
                           for i, pair in enumerate(adapted)], -1)
        scores = torch.cat([cls_head[i](pair[1]).view(bs, self.nc, -1)
                            for i, pair in enumerate(adapted)], -1)
        return dict(boxes=boxes, scores=scores, feats=x)


def replace(model, index, module):
    old = model.model[index]
    for attr in ('i', 'f'):
        setattr(module, attr, getattr(old, attr))
    module.type = type(module).__name__
    module.np = sum(p.numel() for p in module.parameters())
    model.model[index] = module


def build_model(variant='hife', nc=2, seed=20260716, channels=3):
    if variant not in VARIANTS:
        raise ValueError(variant)
    init_seeds(seed, deterministic=True)
    model = DetectionModel('yolo11n.yaml', nc=nc, ch=channels, verbose=False)
    # Shared tensors originate from exactly the same seeded baseline constructor.
    if variant in ('msghost', 'hife'):
        for index in (8, 22):
            old = model.model[index]
            assert isinstance(old, C3k2)
            replace(model, index, MSGhostC3k2(old.cv1.conv.in_channels,
                                             old.cv2.conv.out_channels, n=len(old.m)))
    if variant in ('retention', 'hife'):
        for index in (17, 20):
            old = model.model[index]
            assert isinstance(old, Conv) and old.conv.stride == (2, 2)
            assert old.conv.in_channels == old.conv.out_channels
            replace(model, index, RetentionGSConv(old.conv.in_channels))
    if variant in ('ctd', 'hife'):
        head = model.model[-1]
        assert type(head) is Detect and not head.end2end
        # Same Detect instance preserves every source tensor, bias and inference state.
        head.__class__ = CTDDetect
        head.adapters = nn.ModuleList(CTD(c) for c in (64, 128, 256))
    model.yaml['dfire_variant'] = variant
    return model


def parameter_report(model):
    fused = deepcopy(model).eval().fuse(verbose=False)
    return dict(unfused=sum(p.numel() for p in model.parameters()),
                fused=sum(p.numel() for p in fused.parameters()),
                modules={str(i): sum(p.numel() for p in fused.model[i].parameters())
                         for i in (8, 17, 20, 22)})
