"""Public-release documentation. Scientific logic and parameters are unchanged."""
import os
from pathlib import Path
import json
import math
import random
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
OUT_DIR = Path(__file__).resolve().parents[2] / 'outputs'
DTM_BASE_DIR = OUT_DIR / 'DTM'
DTM_ZBIAS_DIR = OUT_DIR / 'DTM_ZBIAS'
SLOPE_DIR = OUT_DIR / 'SLOPE_ALIGNED'
DIFF_DIR = OUT_DIR / 'DIFF'
LOD95_DIR = OUT_DIR / 'LOD95'
MASK_DIR = OUT_DIR / 'MASK'
ML_IN_DIR = OUT_DIR / 'ML_IN'
ML_CFG_DIR = OUT_DIR / 'ML_CFG'
PNG_PREVIEW_DIR = OUT_DIR / 'PNG_PREVIEW'
ML_IN_DIR.mkdir(parents=True, exist_ok=True)
ML_CFG_DIR.mkdir(parents=True, exist_ok=True)
PNG_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
TAGS = ['230924', '240630', '250816', '251017']
FIRST_TAG = TAGS[0]
PAIR_MODE = 'all'
EXTRA_PAIRS = []
DEM_RES = 0.1
STRICT_SLOPE_PATTERN = True
LABEL_MODE = 'binary'
DIFF_EPS = 0.0
STUDY_MASK_TIF = None
MIN_VALID_RATIO = 0.7
TILE_SIZE = 160
STRIDE = 128
MAX_TILES_PER_PAIR = 0
USE_SLOPE_CHANNEL = True
INCLUDE_DIFF_AS_CHANNEL = False
COMPUTE_STATS = True
STATS_SAMPLE_LIMIT = 200
SAVE_PREVIEW_PNG = False
MAX_PREVIEW_PER_PAIR = 5
VERBOSE = True
RANDOM_SEED = 2025

def log(msg):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if VERBOSE:
        print(msg)

def find_single_file(directory, pattern_list):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    directory = Path(directory)
    for pat in pattern_list:
        candidates = list(directory.glob(pat))
        if candidates:
            if len(candidates) > 1:
                log(f'Public-release status message.{directory} / {pat}Public-release status message.{candidates[0].name}')
            return candidates[0]
    raise FileNotFoundError(f'Public-release status message.{directory}Public-release status message.{pattern_list}')

def make_pairs_from_tags(tags, mode='adjacent', extra_pairs=None):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if mode not in ('adjacent', 'all'):
        raise ValueError('Public-release status message.')
    tags = list(tags)
    pairs = []
    if mode == 'adjacent':
        for i in range(len(tags) - 1):
            pairs.append((tags[i + 1], tags[i]))
    else:
        for i in range(len(tags)):
            for j in range(i):
                pairs.append((tags[i], tags[j]))
    if extra_pairs:
        pairs.extend(extra_pairs)
    pairs = list(dict.fromkeys(pairs))
    return pairs

def open_raster(path):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return rasterio.open(path)

def resolve_dem(tag):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    tag = str(tag)
    if tag == FIRST_TAG:
        patterns = [f'DTM_{tag}_{DEM_RES:.1f}m.tif', f'DTM_{tag}_*.tif']
        return find_single_file(DTM_BASE_DIR, patterns)
    else:
        patterns = [f'DTM_{tag}_zbias_adj_{DEM_RES:.1f}m.tif', f'DTM_{tag}_zbias_adj_*.tif', f'*{tag}*zbias*.tif']
        return find_single_file(DTM_ZBIAS_DIR, patterns)

def resolve_slope(tag):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if STRICT_SLOPE_PATTERN:
        patterns = [f'SLOPE_{tag}_{DEM_RES:.1f}m*.tif', f'*{tag}*.tif']
    else:
        patterns = [f'*{tag}*.tif']
    return find_single_file(SLOPE_DIR, patterns)

def resolve_lod95_sig(later, first):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    patterns = [f'LOD95_sig_{later}-{first}.tif', f'LOD95_sig_{later}-{first}_*.tif']
    return find_single_file(LOD95_DIR, patterns)

def resolve_diff(later, first):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    patterns = [f'DIFF_DTM_{later}-{first}_{DEM_RES:.1f}m*.tif', f'DIFF_DTM_{later}-{first}*.tif', f'*{later}-{first}*.tif']
    return find_single_file(DIFF_DIR, patterns)

def resolve_study_mask():
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if STUDY_MASK_TIF is None:
        return None
    return open_raster(STUDY_MASK_TIF)

def check_grid_compat(base_ds, *others):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    for ds in others:
        if ds is None:
            continue
        if ds.width != base_ds.width or ds.height != base_ds.height:
            raise RuntimeError('Public-release status message.')
        if ds.transform != base_ds.transform:
            raise RuntimeError('Public-release status message.')
        if ds.crs != base_ds.crs:
            raise RuntimeError('Public-release status message.')

def make_label_binary(lod_tile, valid_mask):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    label = np.zeros_like(lod_tile, dtype=np.uint8)
    label[(lod_tile == 1) & valid_mask] = 1
    return label

def make_label_ternary(lod_tile, diff_tile, valid_mask, eps):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    label = np.zeros_like(lod_tile, dtype=np.uint8)
    sig = (lod_tile == 1) & valid_mask
    label[sig & (diff_tile < -eps)] = 1
    label[sig & (diff_tile > eps)] = 2
    return label

def compute_channel_stats(sample_files):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    sums = {}
    sums_sq = {}
    counts = {}
    for npz_path in sample_files:
        d = np.load(npz_path)
        valid = d['valid'].astype(bool)
        for ch in ['dem_t1', 'dem_t2', 'slope_t1', 'slope_t2']:
            if ch not in d:
                continue
            arr = d[ch]
            arr_valid = arr[valid]
            if arr_valid.size == 0:
                continue
            if ch not in sums:
                sums[ch] = 0.0
                sums_sq[ch] = 0.0
                counts[ch] = 0
            sums[ch] += float(arr_valid.sum())
            sums_sq[ch] += float((arr_valid ** 2).sum())
            counts[ch] += int(arr_valid.size)
    stats = {'channels': [], 'mean': {}, 'std': {}}
    for ch in sorted(counts.keys()):
        n = counts[ch]
        m = sums[ch] / n
        v = max(sums_sq[ch] / n - m * m, 0.0)
        s = math.sqrt(v)
        stats['channels'].append(ch)
        stats['mean'][ch] = m
        stats['std'][ch] = s
    return stats

def save_stats_json(stats, path):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

def plot_tile_preview(dem_t1, dem_t2, label, out_path, title=''):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    (fig, axes) = plt.subplots(1, 3, figsize=(9, 3))
    im0 = axes[0].imshow(dem_t1, cmap='terrain')
    axes[0].set_title('DEM t1')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(dem_t2, cmap='terrain')
    axes[1].set_title('DEM t2')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(label, cmap='viridis', vmin=0, vmax=2)
    axes[2].set_title('Label')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def process_one_pair(later, first, global_tile_list, preview_counter):
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    'Public-release status message.'
    log(f'Public-release status message.{later}, first={first}')
    dem_first_path = resolve_dem(first)
    dem_later_path = resolve_dem(later)
    ds_dem_first = open_raster(dem_first_path)
    ds_dem_later = open_raster(dem_later_path)
    ds_slope_first = None
    ds_slope_later = None
    if USE_SLOPE_CHANNEL:
        ds_slope_first = open_raster(resolve_slope(first))
        ds_slope_later = open_raster(resolve_slope(later))
    try:
        lod_path = resolve_lod95_sig(later, first)
        ds_lod = open_raster(lod_path)
    except FileNotFoundError as e:
        log(f'Public-release status message.{later}-{first}Public-release status message.{e}')
        return preview_counter
    ds_diff = None
    if LABEL_MODE == 'ternary' or INCLUDE_DIFF_AS_CHANNEL:
        try:
            diff_path = resolve_diff(later, first)
            ds_diff = open_raster(diff_path)
        except FileNotFoundError as e:
            log(f'Public-release status message.{later}-{first}Public-release status message.{e}')
            return preview_counter
    ds_mask = resolve_study_mask()
    check_grid_compat(ds_dem_first, ds_dem_later, ds_slope_first, ds_slope_later, ds_lod, ds_diff, ds_mask)
    width = ds_dem_first.width
    height = ds_dem_first.height
    transform = ds_dem_first.transform
    crs_wkt = ds_dem_first.crs.to_wkt() if ds_dem_first.crs is not None else ''
    log(f'Public-release status message.{width}, height={height}')
    log(f'[INFO] TILE_SIZE={TILE_SIZE}, STRIDE={STRIDE}, MIN_VALID_RATIO={MIN_VALID_RATIO}')
    if width < TILE_SIZE or height < TILE_SIZE:
        log(f'Public-release status message.{later}-{first}Public-release status message.{TILE_SIZE}Public-release status message.{width}, height={height}Public-release status message.')
        return preview_counter
    tile_count = 0
    saved_preview = preview_counter.get((later, first), 0)
    for y0 in range(0, height - TILE_SIZE + 1, STRIDE):
        for x0 in range(0, width - TILE_SIZE + 1, STRIDE):
            window = Window(x0, y0, TILE_SIZE, TILE_SIZE)
            dem_t1 = ds_dem_first.read(1, window=window, masked=True).filled(np.nan).astype(np.float32)
            dem_t2 = ds_dem_later.read(1, window=window, masked=True).filled(np.nan).astype(np.float32)
            valid = np.isfinite(dem_t1) & np.isfinite(dem_t2)
            if ds_mask is not None:
                mask_tile = ds_mask.read(1, window=window)
                valid = valid & (mask_tile > 0)
            valid_ratio = float(valid.mean())
            if valid_ratio < MIN_VALID_RATIO:
                continue
            slope_t1 = None
            slope_t2 = None
            if USE_SLOPE_CHANNEL:
                slope_t1 = ds_slope_first.read(1, window=window, masked=True).filled(np.nan).astype(np.float32)
                slope_t2 = ds_slope_later.read(1, window=window, masked=True).filled(np.nan).astype(np.float32)
                slope_t1[~valid] = 0.0
                slope_t2[~valid] = 0.0
            lod_tile = ds_lod.read(1, window=window).astype(np.uint8)
            diff_tile = None
            if LABEL_MODE == 'ternary' or INCLUDE_DIFF_AS_CHANNEL:
                diff_tile = ds_diff.read(1, window=window, masked=True).filled(np.nan).astype(np.float32)
                diff_tile[~valid] = 0.0
            if LABEL_MODE == 'binary':
                label = make_label_binary(lod_tile, valid)
            elif LABEL_MODE == 'ternary':
                if diff_tile is None:
                    raise RuntimeError('Public-release status message.')
                label = make_label_ternary(lod_tile, diff_tile, valid, DIFF_EPS)
            else:
                raise ValueError('Public-release status message.')
            dem_t1[~valid] = 0.0
            dem_t2[~valid] = 0.0
            tile_id = f'{later}-{first}_x{x0}_y{y0}_ts{TILE_SIZE}'
            npz_path = ML_IN_DIR / f'{tile_id}.npz'
            np.savez_compressed(npz_path, dem_t1=dem_t1, dem_t2=dem_t2, slope_t1=slope_t1 if slope_t1 is not None else np.zeros_like(dem_t1, dtype=np.float32), slope_t2=slope_t2 if slope_t2 is not None else np.zeros_like(dem_t2, dtype=np.float32), label=label.astype(np.uint8), valid=valid.astype(np.uint8), affine=np.array(transform)[:6].astype(np.float64), crs_wkt=crs_wkt)
            global_tile_list.append({'pair': f'{later}-{first}', 'later': later, 'first': first, 'x': x0, 'y': y0, 'w': TILE_SIZE, 'h': TILE_SIZE, 'valid_ratio': valid_ratio, 'npz': str(npz_path)})
            tile_count += 1
            if SAVE_PREVIEW_PNG and saved_preview < MAX_PREVIEW_PER_PAIR:
                preview_path = PNG_PREVIEW_DIR / f'preview_{tile_id}.png'
                plot_tile_preview(dem_t1, dem_t2, label, preview_path, title=f'{later}-{first}  x={x0},y={y0}')
                saved_preview += 1
            if MAX_TILES_PER_PAIR > 0 and tile_count >= MAX_TILES_PER_PAIR:
                log(f'Public-release status message.{MAX_TILES_PER_PAIR}Public-release status message.')
                break
        if MAX_TILES_PER_PAIR > 0 and tile_count >= MAX_TILES_PER_PAIR:
            break
    log(f'Public-release status message.{later}-{first}Public-release status message.{tile_count}')
    preview_counter[later, first] = saved_preview
    return preview_counter

def main():
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    pairs = make_pairs_from_tags(TAGS, mode=PAIR_MODE, extra_pairs=EXTRA_PAIRS)
    log(f'Public-release status message.{TAGS}')
    log(f'Public-release status message.{len(pairs)}Public-release status message.{pairs}')
    global_tile_list = []
    preview_counter = {}
    for (later, first) in pairs:
        preview_counter = process_one_pair(later, first, global_tile_list, preview_counter)
    if len(global_tile_list) == 0:
        log('Public-release status message.')
        return
    df = pd.DataFrame(global_tile_list)
    index_csv = ML_CFG_DIR / 'tiles_index.csv'
    df.to_csv(index_csv, index=False, encoding='utf-8')
    log(f'Public-release status message.{index_csv}Public-release status message.{len(df)}Public-release status message.')
    if COMPUTE_STATS:
        all_npz_files = df['npz'].tolist()
        if STATS_SAMPLE_LIMIT is not None and STATS_SAMPLE_LIMIT > 0 and (len(all_npz_files) > STATS_SAMPLE_LIMIT):
            sample_files = random.sample(all_npz_files, STATS_SAMPLE_LIMIT)
            log(f'Public-release status message.{STATS_SAMPLE_LIMIT}Public-release status message.{len(all_npz_files)}Public-release status message.')
        else:
            sample_files = all_npz_files
            log(f'Public-release status message.{len(all_npz_files)}Public-release status message.')
        stats = compute_channel_stats(sample_files)
        stats_json = ML_CFG_DIR / 'stats.json'
        save_stats_json(stats, stats_json)
        log(f'Public-release status message.{stats_json}')
        log(f'Public-release status message.{json.dumps(stats, indent=2, ensure_ascii=False)}')
    else:
        log('Public-release status message.')
    log('Public-release status message.')
if __name__ == '__main__':
    main()
