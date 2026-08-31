"""Public-release documentation. Scientific logic and parameters are unchanged."""
import os
import logging
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
OUT_DIR = Path(__file__).resolve().parents[2] / 'outputs'
ML_CFG_DIR = OUT_DIR / 'ML_CFG'
ML_OUT_DIR = OUT_DIR / 'ML_OUT'
SCM_PROB_DIR = OUT_DIR / 'SCM_PROB'
SCM_BIN_DIR = OUT_DIR / 'SCM_BIN'
SCM_PNG_DIR = OUT_DIR / 'SCM_PNG'
TILES_INDEX_CSV = ML_CFG_DIR / 'tiles_index.csv'
for d in [SCM_PROB_DIR, SCM_BIN_DIR, SCM_PNG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
TILE_SIZE: int = 160
STRIDE: int = 128
MERGE_MODE = 'mean'
PROB_THRESH: float = 0.5
NODATA_VALUE: float = np.nan
MIN_TILE_COUNT: int = 1
LOG_TO_FILE: bool = True
LOG_LEVEL: str = 'INFO'
ALLOW_OVERWRITE: bool = True
ENABLE_QUICKLOOK: bool = False
MAX_QUICKLOOK_PAIRS: int = 20

def setup_logging() -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    handlers = []
    stream_handler = logging.StreamHandler()
    handlers.append(stream_handler)
    if LOG_TO_FILE:
        log_file = SCM_PROB_DIR / '6-Siamese_merge_fusion.log'
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        handlers.append(file_handler)
    logging.basicConfig(level=level, format='[%(asctime)s] [%(levelname)s] %(message)s', handlers=handlers)
    logging.info('===== [6] Siamese Merge & Fusion Script Started =====')

def load_tiles_index(csv_path: Path) -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not csv_path.exists():
        raise FileNotFoundError(f'Public-release status message.{csv_path}')
    df = pd.read_csv(csv_path)
    required_cols = ['pair', 'later', 'first', 'x', 'y', 'w', 'h', 'npz']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Public-release status message.{col}')
    df['tile_id'] = df['npz'].apply(lambda p: Path(p).stem)
    logging.info(f"Public-release status message.{len(df)}Public-release status message.{df['pair'].nunique()}Public-release status message.")
    return df

def compute_canvas_shape(df: pd.DataFrame) -> Tuple[int, int]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    max_x1 = (df['x'] + df['w']).max()
    max_y1 = (df['y'] + df['h']).max()
    width = int(max_x1)
    height = int(max_y1)
    logging.info(f'Public-release status message.{width}, height={height}')
    return (height, width)

def load_geo_ref_from_any_npz(df: pd.DataFrame) -> Dict:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    any_npz = Path(df.iloc[0]['npz'])
    if not any_npz.exists():
        raise FileNotFoundError(f'Public-release status message.{any_npz}')
    data = np.load(any_npz)
    if 'affine' not in data or 'crs_wkt' not in data:
        raise ValueError('Public-release status message.')
    affine_arr = data['affine']
    crs_wkt = str(data['crs_wkt'])
    transform = Affine(*affine_arr)
    if crs_wkt and crs_wkt.strip():
        crs = CRS.from_wkt(crs_wkt)
    else:
        crs = None
    logging.info(f'Public-release status message.{any_npz.name}Public-release status message.')
    return {'transform': transform, 'crs': crs}

def write_geotiff_single_band(out_path: Path, data: np.ndarray, transform: Affine, crs: CRS, nodata_value: float, dtype: str='float32') -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if out_path.exists() and (not ALLOW_OVERWRITE):
        raise FileExistsError(f'Public-release status message.{out_path}')
    data_to_write = data.astype(dtype, copy=False)[np.newaxis, :, :]
    profile = {'driver': 'GTiff', 'height': data.shape[0], 'width': data.shape[1], 'count': 1, 'dtype': dtype, 'transform': transform, 'crs': crs, 'nodata': nodata_value, 'compress': 'lzw'}
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data_to_write)
    logging.info(f'Public-release status message.{out_path}')

def quicklook_png(data: np.ndarray, out_path: Path, title: str='') -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not HAS_MPL:
        logging.warning('Public-release status message.')
        return
    if out_path.exists() and (not ALLOW_OVERWRITE):
        logging.warning(f'Public-release status message.{out_path}')
        return
    plt.figure(figsize=(6, 5))
    unique_vals = np.unique(data[np.isfinite(data)])
    if np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [0, 1]):
        im = plt.imshow(data, cmap='gray', vmin=0, vmax=1)
    else:
        vmin = float(np.nanpercentile(data, 2.0))
        vmax = float(np.nanpercentile(data, 98.0))
        if vmax <= vmin:
            (vmin, vmax) = (0.0, 1.0)
        im = plt.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(im, fraction=0.035, pad=0.02)
    plt.title(title, fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    logging.info(f'Public-release status message.{out_path}')

def build_scm_for_pair(df_pair: pd.DataFrame, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    prob_sum = np.zeros((height, width), dtype=np.float32)
    prob_max = np.full((height, width), -np.inf, dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for (idx, row) in df_pair.iterrows():
        x0 = int(row['x'])
        y0 = int(row['y'])
        w = int(row['w'])
        h = int(row['h'])
        x1 = x0 + w
        y1 = y0 + h
        tile_id = str(row['tile_id'])
        prob_path_npy = ML_OUT_DIR / f'prob_{tile_id}.npy'
        prob_path_npz = ML_OUT_DIR / f'prob_{tile_id}.npz'
        prob_map = None
        if prob_path_npy.exists():
            prob_map = np.load(prob_path_npy).astype(np.float32)
        elif prob_path_npz.exists():
            data_npz = np.load(prob_path_npz)
            if 'prob' not in data_npz:
                logging.warning(f'[PAIR] {prob_path_npz.name}Public-release status message.')
                continue
            prob_map = data_npz['prob'].astype(np.float32)
        else:
            logging.warning(f"Public-release status message.{tile_id}Public-release status message.{row['pair']}Public-release status message.")
            continue
        if prob_map.shape != (h, w):
            logging.warning(f'Public-release status message.{tile_id}, prob_shape={prob_map.shape}, (h,w)=({h},{w}Public-release status message.')
            hh = min(h, prob_map.shape[0])
            ww = min(w, prob_map.shape[1])
            prob_map = prob_map[:hh, :ww]
            x1 = x0 + ww
            y1 = y0 + hh
        patch_sum = prob_sum[y0:y1, x0:x1]
        patch_max = prob_max[y0:y1, x0:x1]
        patch_cnt = count[y0:y1, x0:x1]
        if MERGE_MODE == 'mean':
            patch_sum += prob_map
            patch_cnt += 1.0
        elif MERGE_MODE == 'max':
            mask_valid = np.isfinite(prob_map)
            patch_max[mask_valid] = np.where(mask_valid, np.maximum(patch_max[mask_valid], prob_map[mask_valid]), patch_max[mask_valid])
            patch_cnt[mask_valid] += 1.0
        else:
            raise ValueError(f'Public-release status message.{MERGE_MODE}')
    prob_full = np.full((height, width), NODATA_VALUE, dtype=np.float32)
    if MERGE_MODE == 'mean':
        mask = count >= float(MIN_TILE_COUNT)
        if np.any(mask):
            prob_full[mask] = (prob_sum[mask] / count[mask]).astype(np.float32)
    elif MERGE_MODE == 'max':
        mask = count >= float(MIN_TILE_COUNT)
        if np.any(mask):
            prob_full[mask] = prob_max[mask].astype(np.float32)
    bin_full = np.zeros((height, width), dtype=np.uint8)
    valid_prob_mask = np.isfinite(prob_full)
    bin_full[valid_prob_mask & (prob_full >= PROB_THRESH)] = 1
    return (prob_full, bin_full)

def main() -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    setup_logging()
    df = load_tiles_index(TILES_INDEX_CSV)
    unique_w = df['w'].unique()
    unique_h = df['h'].unique()
    if len(unique_w) == 1 and int(unique_w[0]) != TILE_SIZE:
        logging.warning(f'Public-release status message.{unique_w[0]}Public-release status message.{TILE_SIZE}Public-release status message.')
    if len(unique_h) == 1 and int(unique_h[0]) != TILE_SIZE:
        logging.warning(f'Public-release status message.{unique_h[0]}Public-release status message.{TILE_SIZE}Public-release status message.')
    (height, width) = compute_canvas_shape(df)
    geo = load_geo_ref_from_any_npz(df)
    transform = geo['transform']
    crs = geo['crs']
    pairs = sorted(df['pair'].unique())
    logging.info(f'Public-release status message.{len(pairs)}Public-release status message.{pairs}')
    quicklook_count = 0
    for pair in pairs:
        df_pair = df[df['pair'] == pair].copy()
        logging.info(f'Public-release status message.{pair}Public-release status message.{len(df_pair)}')
        (prob_full, bin_full) = build_scm_for_pair(df_pair, height, width)
        out_prob_tif = SCM_PROB_DIR / f'SCM_prob_{pair}.tif'
        out_bin_tif = SCM_BIN_DIR / f'SCM_bin_{pair}_th{PROB_THRESH:.2f}.tif'
        write_geotiff_single_band(out_path=out_prob_tif, data=prob_full, transform=transform, crs=crs, nodata_value=NODATA_VALUE, dtype='float32')
        write_geotiff_single_band(out_path=out_bin_tif, data=bin_full, transform=transform, crs=crs, nodata_value=0, dtype='uint8')
        if ENABLE_QUICKLOOK and quicklook_count < MAX_QUICKLOOK_PAIRS:
            prob_png = SCM_PNG_DIR / f'SCM_prob_{pair}.png'
            bin_png = SCM_PNG_DIR / f'SCM_bin_{pair}_th{PROB_THRESH:.2f}.png'
            quicklook_png(data=prob_full, out_path=prob_png, title=f'SCM Prob {pair}')
            quicklook_png(data=bin_full, out_path=bin_png, title=f'SCM Binary {pair} (th={PROB_THRESH})')
            quicklook_count += 1
    logging.info('Public-release status message.')
if __name__ == '__main__':
    main()
