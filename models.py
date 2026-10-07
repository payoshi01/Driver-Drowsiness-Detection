import torch, torch.nn as nn, torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules
from config import NUM_CLASSES, NUM_FEATURES, QENGINE


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class SmallCNN(nn.Module):
    out_dim = 64

    def __init__(self):
        super().__init__()
        self.quant, self.dequant = QuantStub(), DeQuantStub()
        self.features = nn.Sequential(ConvBNReLU(3, 16, 2), ConvBNReLU(16, 32, 2), ConvBNReLU(32, 64, 2), ConvBNReLU(64, 64, 2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc, self.relu = nn.Linear(64, 64), nn.ReLU()

    def forward(self, x):
        x = self.features(self.quant(x))
        x = torch.flatten(self.pool(x), 1)
        return self.dequant(self.relu(self.fc(x)))

    def fuse(self):                      
        for m in [m for m in self.modules() if isinstance(m, ConvBNReLU)]:
            fuse_modules(m, [["0", "1", "2"]], inplace=True)


class MNv3Encoder(nn.Module):
    out_dim = 64

    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.features, self.pool, self.proj = m.features, nn.AdaptiveAvgPool2d(1), nn.Linear(576, 64)

    def forward(self, x):                
        return F.relu(self.proj(torch.flatten(self.pool(self.features(x)), 1)))


class Fusion(nn.Module):
    def __init__(self, backbone="small", use_crops=True, use_geo=True, pretrained=False):
        super().__init__()
        self.backbone, self.use_crops, self.use_geo = backbone, use_crops, use_geo
        self.register_buffer("f_mean", torch.zeros(NUM_FEATURES))
        self.register_buffer("f_std", torch.ones(NUM_FEATURES))
        d = 0
        if use_crops:
            mk = SmallCNN if backbone == "small" else (lambda: MNv3Encoder(pretrained))
            self.enc_eye, self.enc_mouth = mk(), mk(); d += 128
        if use_geo:
            self.geo = nn.Sequential(nn.Linear(NUM_FEATURES, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()); d += 32
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, NUM_CLASSES))

    def forward(self, eye, mouth, feats):
        parts = []
        if self.use_crops:
            parts += [self.enc_eye((eye - 0.5) / 0.5), self.enc_mouth((mouth - 0.5) / 0.5)]
        if self.use_geo:
            parts.append(self.geo((feats - self.f_mean) / self.f_std))
        return self.head(torch.cat(parts, 1))

MODELS = {
    "geo":         dict(backbone="small", use_crops=False, use_geo=True),
    "cnn":         dict(backbone="small", use_crops=True,  use_geo=False),
    "fusion":      dict(backbone="small", use_crops=True,  use_geo=True),
    "mnv3":        dict(backbone="mnv3",  use_crops=True,  use_geo=False),
    "fusion_mnv3": dict(backbone="mnv3",  use_crops=True,  use_geo=True),
}


def build_model(name, pretrained=False):
    return Fusion(pretrained=pretrained, **MODELS[name])


def load_model(path):
    torch.backends.quantized.engine = QENGINE
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        model = build_model(obj["name"]); model.load_state_dict(obj["state_dict"])
    else:
        model = obj
    return model.eval()