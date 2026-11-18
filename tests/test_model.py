from copy import deepcopy
import torch
from ultralytics import YOLO
from dfire.model import build_model, parameter_report
from dfire.modules import CTD


def test_paper_parameter_budgets():
    baseline = build_model('baseline')
    hife = build_model('hife')
    assert parameter_report(baseline)['unfused'] == 2590230
    report = parameter_report(hife)
    assert report['unfused'] == 2310053
    assert report['fused'] == 2302621
    assert report['modules'] == {'8': 246726, '17': 19296, '20': 75456, '22': 279494}
    assert [sum(p.numel() for p in m.parameters()) for m in hife.model[-1].adapters] == [993, 1889, 3681]


def test_ctd_prediction_identity_and_shared_initialization():
    source = build_model('baseline').eval()
    adapted = build_model('ctd').eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(source(x)[0], adapted(x)[0], rtol=0, atol=0)
    hife = build_model('hife')
    for key, tensor in source.state_dict().items():
        if key.split('.')[1] not in {'8', '17', '20', '22', '23'}:
            torch.testing.assert_close(tensor, hife.state_dict()[key], rtol=0, atol=0)


def test_residual_bound_and_gradients():
    adapter = CTD(64)
    torch.nn.init.normal_(adapter.box_out.weight)
    torch.nn.init.normal_(adapter.cls_out.weight)
    x = torch.randn(2, 64, 8, 8, requires_grad=True)
    box, cls = adapter(x)
    assert (box - x).abs().max() <= .200001
    assert (cls - x).abs().max() <= .200001
    (box.square().mean() + cls.square().mean()).backward()
    assert x.grad.isfinite().all()


def test_detection_loss_backward_and_checkpoint(tmp_path):
    model = build_model('hife').train()
    batch = dict(img=torch.rand(2, 3, 64, 64), batch_idx=torch.tensor([0., 1.]),
                 cls=torch.tensor([[0.], [1.]]), bboxes=torch.tensor([[.5, .5, .3, .3]] * 2))
    from ultralytics.cfg import get_cfg
    model.args = get_cfg()
    loss, _ = model(batch)
    loss.sum().backward()
    assert model.model[-1].adapters[0].box_out.weight.grad.isfinite().all()
    path = tmp_path / 'hife.pt'
    torch.save({'model': deepcopy(model).half(), 'train_args': {'task': 'detect'}}, path)
    loaded = YOLO(path).model.float().eval()
    model.eval()
    with torch.no_grad():
        # Serialized half precision introduces expected rounding.
        assert loaded(batch['img'])[0].shape == model(batch['img'])[0].shape
        assert loaded.yaml['dfire_variant'] == 'hife'
