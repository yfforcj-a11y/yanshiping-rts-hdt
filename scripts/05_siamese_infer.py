from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
OUT_DIR = Path(__file__).resolve().parents[2] / 'outputs'
ML_IN_DIR = OUT_DIR / 'ML_IN'
ML_CFG_DIR = OUT_DIR / 'ML_CFG'
ML_MODEL_DIR = OUT_DIR / 'ML_MODEL'
ML_OUT_DIR = OUT_DIR / 'ML_OUT'
ML_OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CHECKPOINT = ML_MODEL_DIR / 'siamese_unet_epoch8.pth'
BASE_FILTERS = 32
USE_BN = True
DROPOUT_P = 0.0
BATCH_SIZE = 8
NUM_WORKERS = 0

def load_channel_stats(stats_path: Path):
    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    channels = stats['channels']
    mean = np.array([stats['mean'][c] for c in channels], dtype=np.float32)
    std = np.array([stats['std'][c] for c in channels], dtype=np.float32)
    std = np.where(std > 1e-06, std, 1.0)
    return (mean, std)

class InferDataset(Dataset):

    def __init__(self, npz_dir: Path, mean: np.ndarray, std: np.ndarray):
        self.files = sorted(npz_dir.glob('*.npz'))
        if not self.files:
            raise RuntimeError(f'No .npz files found under {npz_dir}')
        self.mean = mean.reshape(-1, 1, 1)
        self.std = std.reshape(-1, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        with np.load(p) as d:
            x = np.stack([d['dem_t1'], d['dem_t2'], d['slope_t1'], d['slope_t2']], axis=0).astype(np.float32)
            valid = d['valid'].astype(np.float32)
        x = (x - self.mean) / self.std
        return (torch.from_numpy(x), torch.from_numpy(valid), p.stem)

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, use_bn=True):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        layers = [nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, use_bn=use_bn))

    def forward(self, x):
        return self.block(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_bn=use_bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):

    def __init__(self, in_channels, base_filters, use_bn=True):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_filters, use_bn=use_bn)
        self.down1 = Down(base_filters, base_filters * 2, use_bn=use_bn)
        self.down2 = Down(base_filters * 2, base_filters * 4, use_bn=use_bn)
        self.down3 = Down(base_filters * 4, base_filters * 8, use_bn=use_bn)
        self.down4 = Down(base_filters * 8, base_filters * 8, use_bn=use_bn)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return (x1, x2, x3, x4, x5)

class SiameseUNet(nn.Module):

    def __init__(self, base_filters=32, use_bn=True, dropout_p=0.0):
        super().__init__()
        self.encoder = Encoder(2, base_filters, use_bn=use_bn)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        self.up1 = Up(base_filters * 16, base_filters * 4, use_bn=use_bn)
        self.up2 = Up(base_filters * 8, base_filters * 2, use_bn=use_bn)
        self.up3 = Up(base_filters * 4, base_filters, use_bn=use_bn)
        self.up4 = Up(base_filters * 2, base_filters, use_bn=use_bn)
        self.outc = OutConv(base_filters)

    def forward(self, x):
        x_t1 = torch.stack([x[:, 0, :, :], x[:, 2, :, :]], dim=1)
        x_t2 = torch.stack([x[:, 1, :, :], x[:, 3, :, :]], dim=1)
        (a1, a2, a3, a4, a5) = self.encoder(x_t1)
        (b1, b2, b3, b4, b5) = self.encoder(x_t2)
        d1 = torch.abs(b1 - a1)
        d2 = torch.abs(b2 - a2)
        d3 = torch.abs(b3 - a3)
        d4 = torch.abs(b4 - a4)
        d5 = self.dropout(torch.abs(b5 - a5))
        x = self.up1(d5, d4)
        x = self.up2(x, d3)
        x = self.up3(x, d2)
        x = self.up4(x, d1)
        return self.outc(x)

def main():
    if not MODEL_CHECKPOINT.exists():
        raise FileNotFoundError(f'Missing checkpoint: {MODEL_CHECKPOINT}')
    (mean, std) = load_channel_stats(ML_CFG_DIR / 'stats.json')
    dataset = InferDataset(ML_IN_DIR, mean, std)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[MAIN] device={device}')
    model = SiameseUNet(BASE_FILTERS, USE_BN, DROPOUT_P).to(device)
    state = torch.load(MODEL_CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        for (x, valid, stems) in tqdm(loader, leave=False):
            x = x.to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy()[:, 0, :, :]
            valid = valid.numpy()
            for (i, stem) in enumerate(stems):
                out = np.where(valid[i] > 0, prob[i], np.nan).astype(np.float32)
                np.save(ML_OUT_DIR / f'prob_{stem}.npy', out)
    print(f'[MAIN] inference complete, outputs saved to {ML_OUT_DIR}')
if __name__ == '__main__':
    main()
