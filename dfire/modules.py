"""HIFE-YOLO equations 1–13: https://doi.org/10.3390/fire9110478."""
import math
import torch
from torch import nn
from ultralytics.nn.modules import Conv


class MSGhostConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        assert c2 % 2 == 0
        h = c2 // 2
        self.primary = Conv(c1, h, 3)
        self.branches = nn.ModuleList(Conv(h, h, k, g=h) for k in (3, 5, 7))
        self.gate = nn.Linear(h, 3)

    def forward(self, x):
        y = self.primary(x)
        weights = self.gate(y.mean((2, 3))).softmax(1)
        z = sum(branch(y) * weights[:, i, None, None, None]
                for i, branch in enumerate(self.branches))
        return torch.cat((y, z), 1)


class MSGhostBottleneck(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(MSGhostConv(c, c // 2), MSGhostConv(c // 2, c))

    def forward(self, x):
        return x + self.body(x)


class MSGhostC3k2(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        h = int(c2 * e)
        self.cv1 = Conv(c1, 2 * h, 1)
        self.blocks = nn.ModuleList(MSGhostBottleneck(h) for _ in range(n))
        self.cv2 = Conv((2 + n) * h, c2, 1)

    def forward(self, x):
        ys = list(self.cv1(x).chunk(2, 1))
        for block in self.blocks:
            ys.append(block(ys[-1]))
        return self.cv2(torch.cat(ys, 1))


class RetentionGSConv(nn.Module):
    def __init__(self, c, rho=0.25):
        super().__init__()
        self.primary = Conv(c, c // 2, 3, 2)
        self.cheap = Conv(c // 2, c // 2, 5, g=c // 2)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.rho = rho

    def forward(self, x):
        a = self.primary(x)
        y = torch.cat((a, self.cheap(a)), 1)
        b, c, h, w = y.shape
        y = y.reshape(b, 2, c // 2, h, w).transpose(1, 2).reshape(b, c, h, w)
        return (y + self.rho * self.pool(x)) / math.sqrt(1 + self.rho ** 2)


class CTD(nn.Module):
    def __init__(self, c, hidden=4, eta=0.2):
        super().__init__()
        self.reduce = nn.Conv2d(c, hidden, 1, bias=False)
        self.detail = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.context = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden, bias=False)
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.selector = nn.Conv2d(2 * hidden, 1, 1)
        self.cls_norm = nn.GroupNorm(1, hidden)
        self.box_norm = nn.GroupNorm(1, hidden)
        self.cls_out = nn.Conv2d(hidden, c, 1)
        self.box_out = nn.Conv2d(hidden, c, 1)
        self.eta = eta
        self.reset_projections()

    def reset_projections(self):
        for layer in (self.cls_out, self.box_out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        h = self.reduce(x)
        d, c = self.detail(h) - self.pool(h), self.context(h)
        s = self.selector(torch.cat((d.abs(), c), 1)).sigmoid()
        cls = self.cls_out(torch.nn.functional.silu(self.cls_norm(c + s * d)))
        box = self.box_out(torch.nn.functional.silu(self.box_norm(d + s * c)))
        return x + self.eta * box.tanh(), x + self.eta * cls.tanh()
