from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[2] / 'outputs'
ML_IN_DIR = PROJECT_ROOT / 'ML_IN'
ML_CFG_DIR = PROJECT_ROOT / 'ML_CFG'
MODEL_OUT_DIR = PROJECT_ROOT / 'ML_MODEL'
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_EPOCHS = 8
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-05
VAL_SPLIT = 0.2
RANDOM_SEED = 2025
IN_CHANNELS = 4
BASE_FILTERS = 32
USE_BN = True
DROPOUT_P = 0.0
USE_DICE_LOSS = True
POS_WEIGHT = 2.0
USE_FLIP_AUG = True
USE_ROTATE_AUG = False
NUM_WORKERS = 0
PATCH_H = 160
PATCH_W = 160
SAVE_BEST_ONLY = False
PLOT_TRAIN_CURVES = False

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_channel_stats(stats_path: Path):
    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    channels = stats['channels']
    mean = np.array([stats['mean'][c] for c in channels], dtype=np.float32)
    std = np.array([stats['std'][c] for c in channels], dtype=np.float32)
    std = np.where(std > 1e-06, std, 1.0)
    return (channels, mean, std)

class TileDataset(Dataset):

    def __init__(self, npz_dir: Path, mean: np.ndarray, std: np.ndarray, train: bool):
        self.files = sorted(npz_dir.glob('*.npz'))
        if not self.files:
            raise RuntimeError(f'No .npz files found under {npz_dir}')
        self.mean = mean.reshape(-1, 1, 1)
        self.std = std.reshape(-1, 1, 1)
        self.train = train
        filtered = []
        for p in self.files:
            with np.load(p) as d:
                if d['dem_t1'].shape == (PATCH_H, PATCH_W):
                    filtered.append(p)
        if not filtered:
            raise RuntimeError(f'No {PATCH_H}x{PATCH_W} tiles found under {npz_dir}')
        self.files = filtered

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as d:
            x = np.stack([d['dem_t1'], d['dem_t2'], d['slope_t1'], d['slope_t2']], axis=0).astype(np.float32)
            y = d['label'].astype(np.float32)[None, ...]
            valid = d['valid'].astype(np.float32)[None, ...]
        x = (x - self.mean) / self.std
        if self.train and USE_FLIP_AUG:
            if random.random() < 0.5:
                x = np.flip(x, axis=2).copy()
                y = np.flip(y, axis=2).copy()
                valid = np.flip(valid, axis=2).copy()
            if random.random() < 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=1).copy()
                valid = np.flip(valid, axis=1).copy()
        if self.train and USE_ROTATE_AUG:
            k = random.randint(0, 3)
            if k:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                y = np.rot90(y, k=k, axes=(1, 2)).copy()
                valid = np.rot90(valid, k=k, axes=(1, 2)).copy()
        return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(valid))

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

    def __init__(self, in_channels=4, base_filters=32, use_bn=True, dropout_p=0.0):
        super().__init__()
        self.encoder = Encoder(in_channels, base_filters, use_bn=use_bn)
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

def dice_loss(logits, target, valid_mask, eps=1e-06):
    prob = torch.sigmoid(logits)
    prob = prob * valid_mask
    target = target * valid_mask
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

def compute_loss(logits, target, valid_mask):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none', pos_weight=torch.tensor(POS_WEIGHT, device=logits.device))
    bce = (bce * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    if USE_DICE_LOSS:
        dloss = dice_loss(logits, target, valid_mask)
        return (bce + dloss, bce.item(), dloss.item())
    return (bce, bce.item(), 0.0)

def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    (losses, bces, dices) = ([], [], [])
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for (x, y, valid) in tqdm(loader, leave=False):
            x = x.to(device)
            y = y.to(device)
            valid = valid.to(device)
            logits = model(x)
            (loss, bce, dloss) = compute_loss(logits, y, valid)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            bces.append(bce)
            dices.append(dloss)
    return (float(np.mean(losses)), float(np.mean(bces)), float(np.mean(dices)))

def main():
    set_seed(RANDOM_SEED)
    (_, mean, std) = load_channel_stats(ML_CFG_DIR / 'stats.json')
    dataset = TileDataset(ML_IN_DIR, mean, std, train=True)
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    (train_set, val_set) = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(RANDOM_SEED))
    train_set.dataset.train = True
    val_set.dataset.train = False
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[MAIN] device={device}')
    print(f'[MAIN] train_tiles={len(train_set)}, val_tiles={len(val_set)}')
    model = SiameseUNet(in_channels=2, base_filters=BASE_FILTERS, use_bn=USE_BN, dropout_p=DROPOUT_P).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = {'train_loss': [], 'val_loss': [], 'train_bce': [], 'val_bce': [], 'train_dice': [], 'val_dice': []}
    best_val = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'\n========== Epoch {epoch}/{NUM_EPOCHS} ==========')
        (tr_loss, tr_bce, tr_dice) = run_epoch(model, train_loader, device, optimizer)
        (va_loss, va_bce, va_dice) = run_epoch(model, val_loader, device, optimizer=None)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_bce'].append(tr_bce)
        history['val_bce'].append(va_bce)
        history['train_dice'].append(tr_dice)
        history['val_dice'].append(va_dice)
        print(f'[EPOCH {epoch}] train_loss={tr_loss:.4f}, val_loss={va_loss:.4f}')
        model_path = MODEL_OUT_DIR / f'siamese_unet_epoch{epoch}.pth'
        if SAVE_BEST_ONLY:
            if va_loss < best_val:
                best_val = va_loss
                torch.save(model.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
    with open(MODEL_OUT_DIR / 'train_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f'[MAIN] training complete, outputs saved to {MODEL_OUT_DIR}')
if __name__ == '__main__':
    main()
