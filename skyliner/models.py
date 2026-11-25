"""Networks (PyTorch, training only; inference uses the exported ONNX files).

SceneNet  : 512x384 RGB -> 4-class logits (sky, land, sea, occluder) at 512x384.
            Gives the topology: which boundary is the topmost land/sky edge, where the
            open sea shows, where ship structure hides the view.
EdgeNet   : full-resolution RGB -> 4 boundary heatmap logits (L0 visible sea horizon,
            L1 silhouette, L2 inner ridge, L0 under distant land). Gives sub-pixel position along each column.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SCENE_W, SCENE_H = 512, 384
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def cbr(i, o, s=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, padding=d, dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("m", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):  # x: uint8-range float 0..255
        return (x / 255.0 - self.m) / self.s


class SceneNet(nn.Module):
    def __init__(self, n_cls=4, w=(16, 32, 48, 64, 96)):
        super().__init__()
        self.norm = Normalize()
        self.e0 = nn.Sequential(cbr(3, w[0], 2), cbr(w[0], w[0]))          # /2
        self.e1 = nn.Sequential(cbr(w[0], w[1], 2), cbr(w[1], w[1]))       # /4
        self.e2 = nn.Sequential(cbr(w[1], w[2], 2), cbr(w[2], w[2]))       # /8
        self.e3 = nn.Sequential(cbr(w[2], w[3], 2), cbr(w[3], w[3]))       # /16
        self.e4 = nn.Sequential(cbr(w[3], w[4], 2), cbr(w[4], w[4], d=2), cbr(w[4], w[4], d=4))  # /32
        self.d3 = cbr(w[4] + w[3], w[3])
        self.d2 = cbr(w[3] + w[2], w[2])
        self.d1 = cbr(w[2] + w[1], w[1])
        self.d0 = cbr(w[1] + w[0], w[0])
        self.head = nn.Sequential(cbr(w[0] + 3, 16), nn.Conv2d(16, n_cls, 1))

    @staticmethod
    def up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.norm(x)
        e0 = self.e0(x); e1 = self.e1(e0); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        d = self.d3(torch.cat([self.up(e4, e3), e3], 1))
        d = self.d2(torch.cat([self.up(d, e2), e2], 1))
        d = self.d1(torch.cat([self.up(d, e1), e1], 1))
        d = self.d0(torch.cat([self.up(d, e0), e0], 1))
        return self.head(torch.cat([self.up(d, x), x], 1))


class EdgeNet(nn.Module):
    """Small full-res FCN, receptive field ~60 px, keeps a full-res path for sub-pixel accuracy."""

    def __init__(self, n_out=4, c=24):
        super().__init__()
        self.norm = Normalize()
        self.f0 = nn.Sequential(cbr(3, c), cbr(c, c))
        self.f1 = nn.Sequential(cbr(c, 32, 2), cbr(32, 32), cbr(32, 32, d=2), cbr(32, 32, d=4), cbr(32, 32, d=8))
        self.fuse = nn.Sequential(cbr(c + 32 + 3, c), cbr(c, c))
        self.out = nn.Conv2d(c, n_out, 1)

    def forward(self, x):
        x = self.norm(x)
        a = self.f0(x)
        b = self.f1(a)
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.fuse(torch.cat([a, b, x], 1)))
