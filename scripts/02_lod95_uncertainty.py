"""Public-release documentation. Scientific logic and parameters are unchanged."""
from glob import glob
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import os
import sys
import time
import math
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterio.features import geometry_mask
OUT_DIR = Path(__file__).resolve().parents[2] / 'outputs'
DIFF_DIR = Path(__file__).resolve().parents[2] / 'data' / 'dod_0p1m'
PNG_DIR = OUT_DIR / 'PNG'
LOD_DIR = OUT_DIR / 'LOD95'
ZONE_DIR = OUT_DIR / 'ZONES'
STAT_DIR = OUT_DIR / 'STATS'
UNC_DIR = OUT_DIR / 'UNCERTAINTY'
for d in [LOD_DIR, ZONE_DIR, STAT_DIR, UNC_DIR, PNG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
TAGS = ['230924', '240630', '250816', '251017']
FIRST = TAGS[0]
RES = 0.1
CELL_AREA = RES * RES
MANUAL_PAIRS: List[Tuple[str, str]] = [('240630', '230924'), ('250816', '240630'), ('251017', '250816'), ('250816', '230924'), ('251017', '240630'), ('251017', '230924')]
ADJACENT_CHAIN_PAIRS: List[Tuple[str, str]] = [('240630', '230924'), ('250816', '240630'), ('251017', '250816')]
AUTO_DISCOVER_PAIRS = False
ALLOW_SKIP_MISSING = True
PREFER_RES = RES
DEM_RMSE: Dict[str, float] = {'230924': 0.08, '240630': 0.09, '250816': 0.1, '251017': 0.1}
STUDY_MASK_TIF: Optional[str] = None
SIGMA_DIFF_RASTERS: Dict[str, str] = {}
ZONAL_SHP: Optional[str] = None
ZONAL_NAME_FIELD = 'ZONENAME'
USE_SIGMA_RASTER = True
ONLY_SIGNIFICANT_FOR_VOLUME_UNCERTAINTY = True
WRITE_INTERMEDIATE = True
MAKE_PREVIEW_PNG = True
MAKE_SERIES_PLOT = True
MAKE_ZONAL_STATS = True
PNG_DPI = 600
ENABLE_PLOTTING = False
VERBOSE = True
DIFF_SYM_RANGE = 12.0
CMAP_DIFF_HEX = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffff', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
ZONE_COLORS = {1: '#2E86C1', 2: '#B3B6B7', 3: '#E74C3C'}
RUNLOG_PATH = STAT_DIR / 'run_log.txt'
RUNREP_PATH = STAT_DIR / 'run_report.csv'
LOG_TO_FILE = True

def _ts() -> str:
    return time.strftime('[%Y-%m-%d %H:%M:%S]')

def log(msg: str) -> None:
    line = f'{_ts()} {msg}'
    if VERBOSE:
        print(line)
    if LOG_TO_FILE:
        RUNLOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RUNLOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

def log_section(title: str) -> None:
    bar = '=' * 66
    log(f'{bar}\n{title}\n{bar}')

def log_kv(k: str, v) -> None:
    log(f'{k}={v}')

def pair_key(later: str, first: str) -> str:
    return f'{later}-{first}'

def pair_sort_key(pair: Tuple[str, str]) -> Tuple[int, int]:
    (later, first) = pair
    return (TAGS.index(later), TAGS.index(first))

def is_adjacent_pair(later: str, first: str) -> bool:
    return (later, first) in ADJACENT_CHAIN_PAIRS

def safe_nanpercentile(arr: np.ndarray, q: float, default: float) -> float:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return default
    return float(np.nanpercentile(vals, q))

def parse_px_size(tif: Path) -> float:
    with rasterio.open(tif) as src:
        return float(abs(src.transform.a))

def cmap_from_hex(hex_list: List[str]):
    return mcolors.LinearSegmentedColormap.from_list('cmap', hex_list, N=256)

def read_tif(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        meta = src.meta.copy()
    return (arr, meta)

def save_tif(path: Path, arr, meta: dict, dtype: str, nodata=0):
    meta = meta.copy()
    meta.update(dtype=dtype, nodata=nodata, compress='lzw', tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, 'w', **meta) as dst:
        dst.write(arr.astype(dtype), 1)

def read_mask_bool(path: Optional[str], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if path is None:
        return None
    with rasterio.open(path) as src:
        m = src.read(1)
    if m.shape != shape:
        raise ValueError('Public-release status message.')
    return m.astype(bool)

def find_diff_path(later: str, first: str) -> Optional[Path]:
    release = DIFF_DIR / f'yanshiping_rts_dod_20{later}_minus_20{first}_0p1m.tif'
    if release.exists():
        return release
    strict = DIFF_DIR / f'DIFF_DTM_{later}-{first}_{RES:.1f}m.tif'
    if strict.exists():
        try:
            px = parse_px_size(strict)
        except Exception:
            px = None
        log(f'[DIFF] strict hit later={later} first={first} file={strict.name} px={px}')
        return strict
    candidates = [Path(p) for p in glob(str(DIFF_DIR / f'DIFF_DTM_{later}-{first}_*.tif'))]
    if not candidates:
        log(f'[DIFF] not found later={later} first={first}')
        return None
    scored = []
    for p in candidates:
        try:
            px = parse_px_size(p)
            ref = PREFER_RES if PREFER_RES is not None else px
            scored.append((abs(px - ref), px, p))
        except Exception:
            scored.append((1000000000.0, None, p))
    scored.sort(key=lambda x: (x[0], str(x[2])))
    pick = scored[0]
    log(f'[DIFF] wildcard pick later={later} first={first} file={pick[2].name} px={pick[1]}')
    return pick[2]

def diff_tif(later: str, first: str) -> Optional[Path]:
    return find_diff_path(later, first)

def lod95_const(rmse_first: float, rmse_later: float) -> float:
    return float(1.96 * math.sqrt(rmse_first * rmse_first + rmse_later * rmse_later))

def discover_pairs_from_disk(tags: List[str]) -> List[Tuple[str, str]]:
    files = glob(str(DIFF_DIR / 'DIFF_DTM_*-*_*.tif'))
    seen = set()
    for fp in files:
        name = Path(fp).stem
        try:
            core = name.split('DIFF_DTM_')[1]
            pair = core.split('_')[0]
            (later, first) = pair.split('-')
            seen.add((later, first))
        except Exception:
            continue
    out: List[Tuple[str, str]] = []
    for p in MANUAL_PAIRS:
        if p in seen:
            out.append(p)
    for p in sorted(seen, key=pair_sort_key):
        if p not in out:
            out.append(p)
    log(f'[DISCOVER] pairs={out}')
    return out
if AUTO_DISCOVER_PAIRS:
    PAIRS = discover_pairs_from_disk(TAGS)
else:
    PAIRS = MANUAL_PAIRS.copy()

def build_lod95_and_significance(diff_path: Path, rmse_first: float, rmse_later: float, pair_key_str: str, study_mask_bool: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, dict, bool, float]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    (a, meta) = read_tif(diff_path)
    dz = np.ma.filled(a, np.nan)
    valid = np.isfinite(dz)
    if study_mask_bool is not None:
        valid &= study_mask_bool
    lod_const = lod95_const(rmse_first, rmse_later)
    if USE_SIGMA_RASTER and pair_key_str in SIGMA_DIFF_RASTERS:
        (s, _) = read_tif(Path(SIGMA_DIFF_RASTERS[pair_key_str]))
        sigma = np.ma.filled(s, np.nan)
        if sigma.shape != dz.shape:
            raise ValueError(f'Public-release status message.{pair_key_str}')
        lod95 = 1.96 * sigma
        signif = np.zeros_like(dz, dtype='uint8')
        signif[valid & np.isfinite(lod95) & (np.abs(dz) >= lod95)] = 1
        return (lod95, signif, meta, True, lod_const)
    lod95 = np.full_like(dz, lod_const, dtype='float32')
    signif = np.zeros_like(dz, dtype='uint8')
    signif[valid & (np.abs(dz) >= lod_const)] = 1
    return (lod95, signif, meta, False, lod_const)

def write_morpho_zones(diff_path: Path, signif: np.ndarray, lod95: np.ndarray, is_grid: bool, study_mask_bool: Optional[np.ndarray]) -> Tuple[np.ndarray, dict]:
    (a, meta) = read_tif(diff_path)
    dz = np.ma.filled(a, np.nan)
    valid = np.isfinite(dz)
    if study_mask_bool is not None:
        valid &= study_mask_bool
    Z = np.zeros_like(dz, dtype='uint8')
    if is_grid:
        erosion = valid & np.isfinite(lod95) & (dz <= -lod95)
        stable = valid & np.isfinite(lod95) & (np.abs(dz) < lod95)
        depo = valid & np.isfinite(lod95) & (dz >= +lod95)
        Z[erosion] = 1
        Z[stable] = 2
        Z[depo] = 3
    else:
        Z[valid & (signif == 1) & (dz < 0)] = 1
        Z[valid & (signif == 0)] = 2
        Z[valid & (signif == 1) & (dz > 0)] = 3
    return (Z, meta)

def volume_table(diff_path: Path, zones_arr: Optional[np.ndarray], study_mask_bool: Optional[np.ndarray]) -> pd.DataFrame:
    (a, _) = read_tif(diff_path)
    dz = np.ma.filled(a, np.nan)
    valid = np.isfinite(dz)
    if study_mask_bool is not None:
        valid &= study_mask_bool

    def _one(mask: np.ndarray, name: str) -> Dict[str, float]:
        m = valid & mask
        area = float(m.sum() * CELL_AREA)
        if not m.any():
            return dict(Type=name, Area_m2=0.0, Mean_dz_m=0.0, Volume_m3=0.0)
        return dict(Type=name, Area_m2=area, Mean_dz_m=float(np.nanmean(dz[m])), Volume_m3=float(np.nansum(dz[m]) * CELL_AREA))
    rows: List[Dict[str, float]] = []
    if zones_arr is not None:
        rows.append(_one(zones_arr == 1, 'Erosion'))
        rows.append(_one(zones_arr == 2, 'Stable'))
        rows.append(_one(zones_arr == 3, 'Deposition'))
        total_mask = np.ones_like(dz, dtype=bool)
        rows.append(_one(total_mask, 'Total'))
    else:
        rows.append(_one(np.ones_like(dz, dtype=bool), 'All'))
    return pd.DataFrame(rows)

def volume_uncertainty_sigmaV(signif: np.ndarray, lod95: np.ndarray, is_grid: bool, study_mask_bool: Optional[np.ndarray]) -> float:
    if is_grid:
        sigma = lod95 / 1.96
    else:
        sigma = np.full_like(lod95, float(lod95[0, 0] / 1.96), dtype='float32')
    use = np.isfinite(sigma)
    if ONLY_SIGNIFICANT_FOR_VOLUME_UNCERTAINTY:
        use &= signif == 1
    if study_mask_bool is not None:
        use &= study_mask_bool
    sigmasq_sum = float(np.nansum(np.where(use, sigma, 0.0) ** 2))
    return float(CELL_AREA * math.sqrt(sigmasq_sum))

def centroid_from_zone_array(zones_arr: np.ndarray, transform, code: int) -> Optional[Tuple[float, float]]:
    (ys, xs) = np.where(zones_arr == code)
    if xs.size == 0:
        return None
    X = transform.c + xs * transform.a + 0.5 * transform.a
    Y = transform.f + ys * transform.e + 0.5 * transform.e
    return (float(np.mean(X)), float(np.mean(Y)))

def plot_array(data: np.ndarray, out_png: Path, title: str, cmap, vmin=None, vmax=None, label: Optional[str]=None) -> None:
    if not ENABLE_PLOTTING:
        return
    (fig, ax) = plt.subplots(figsize=(9, 7))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest', resample=False)
    cb = plt.colorbar(im, fraction=0.03, pad=0.02)
    if label:
        cb.set_label(label, fontname='Times New Roman', fontsize=10)
    ax.set_title(title, fontname='Times New Roman', fontsize=12)
    ax.axis('off')
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95)
    plt.savefig(out_png, dpi=PNG_DPI, transparent=True)
    plt.close(fig)

def plot_diff_preview(diff_path: Path, key: str) -> None:
    (a, _) = read_tif(diff_path)
    dz = np.ma.filled(a, np.nan)
    cmap = cmap_from_hex(CMAP_DIFF_HEX)
    plot_array(dz, PNG_DIR / f'DIFF_{key}.png', title=f'DoD {key}', cmap=cmap, vmin=-DIFF_SYM_RANGE, vmax=DIFF_SYM_RANGE, label='Elevation change (m)')

def plot_zone_preview(zones_arr: np.ndarray, key: str) -> None:
    if not ENABLE_PLOTTING:
        return
    vals = [1, 2, 3]
    colors = [ZONE_COLORS[v] for v in vals]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    (fig, ax) = plt.subplots(figsize=(9, 7))
    im = ax.imshow(zones_arr, cmap=cmap, norm=norm, interpolation='nearest', resample=False)
    cbar = plt.colorbar(im, ticks=[1, 2, 3], fraction=0.03, pad=0.02)
    cbar.ax.set_yticklabels(['Erosion', 'Stable', 'Deposition'])
    ax.set_title(f'Zones {key}', fontname='Times New Roman', fontsize=12)
    ax.axis('off')
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95)
    plt.savefig(PNG_DIR / f'ZONES_{key}.png', dpi=PNG_DPI, transparent=True)
    plt.close(fig)

def plot_pair_bar(results: List[Dict[str, Union[str, float]]]) -> None:
    if not ENABLE_PLOTTING:
        return
    if not results:
        return
    labels = [str(r['pair']) for r in results]
    vols = [float(r['Vol_total']) for r in results]
    (fig, ax) = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(labels, vols, color='#2E86C1')
    ax.axhline(0.0, color='k', lw=0.8)
    ax.set_ylabel('Net Volume (m3)', fontname='Times New Roman')
    ax.set_title('Net Volume by Pair', fontname='Times New Roman')
    ax.grid(axis='y', ls='--', alpha=0.35)
    plt.xticks(rotation=25, ha='right')
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.92)
    plt.savefig(PNG_DIR / 'Volume_by_pair_all.png', dpi=PNG_DPI, transparent=True)
    plt.close(fig)

def plot_adjacent_series(results: List[Dict[str, Union[str, float]]]) -> None:
    if not ENABLE_PLOTTING:
        return
    if not results:
        return
    labels = [str(r['pair']) for r in results]
    vols = [float(r['Vol_total']) for r in results]
    (fig, ax) = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(labels, vols, '-o', color='#2E86C1', lw=1.8)
    ax.axhline(0.0, color='k', lw=0.8)
    ax.set_ylabel('Net Volume (m3)', fontname='Times New Roman')
    ax.set_title('Net Volume Time Series (Adjacent Pairs)', fontname='Times New Roman')
    ax.grid(ls='--', alpha=0.35)
    plt.xticks(rotation=20, ha='right')
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.18, top=0.92)
    plt.savefig(PNG_DIR / 'Volume_series_adjacent.png', dpi=PNG_DPI, transparent=True)
    plt.close(fig)

def plot_motion_vectors(moves: List[Dict[str, Union[str, float]]], kind: str, fname: str, color: str) -> None:
    if not ENABLE_PLOTTING:
        return
    (fig, ax) = plt.subplots(figsize=(6.2, 6.2))
    plotted = 0
    for row in moves:
        dx = row.get(f'{kind}_dx')
        dy = row.get(f'{kind}_dy')
        if dx is None or dy is None:
            continue
        ax.arrow(0, 0, dx, dy, head_width=max(0.5, 0.02 * max(abs(dx), abs(dy), 1.0)), fc=color, ec=color, alpha=0.7, length_includes_head=True)
        plotted += 1
    ax.set_title(f'{kind} Centroid Motion (Adjacent Pairs)', fontname='Times New Roman')
    ax.grid(ls=':', alpha=0.4)
    ax.set_aspect('equal')
    if plotted == 0:
        ax.text(0.5, 0.5, 'No valid motion vectors', ha='center', va='center', transform=ax.transAxes)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92)
    plt.savefig(PNG_DIR / fname, dpi=PNG_DPI, transparent=True)
    plt.close(fig)

def run_one_pair(later: str, first: str) -> Dict[str, Union[str, float, bool, Optional[Tuple[float, float]]]]:
    key = pair_key(later, first)
    log_section(f'RUN PAIR {key}')
    diff = diff_tif(later, first)
    if diff is None or not Path(diff).exists():
        if ALLOW_SKIP_MISSING:
            log(f'[SKIP] missing diff for {key}')
            return {}
        raise FileNotFoundError(DIFF_DIR / f'DIFF_DTM_{later}-{first}_*.tif')
    try:
        px = parse_px_size(diff)
    except Exception:
        px = None
    log_kv('diff_file', diff.name)
    log_kv('pixel_size(m)', px)
    (a, meta) = read_tif(diff)
    study_mask_bool = read_mask_bool(STUDY_MASK_TIF, a.shape)
    if study_mask_bool is not None:
        log_kv('study_mask', 'on')
    s1 = DEM_RMSE[first]
    s2 = DEM_RMSE[later]
    log_kv('RMSE_first', s1)
    log_kv('RMSE_later', s2)
    (lod95_grid, signif, meta, is_grid, lod95_const_val) = build_lod95_and_significance(diff, s1, s2, key, study_mask_bool)
    log_kv('LOD95_mode', 'grid' if is_grid else 'const')
    log_kv('LOD95_const(m)', lod95_const_val)
    (zones_arr, z_meta) = write_morpho_zones(diff, signif, lod95_grid, is_grid, study_mask_bool)
    if WRITE_INTERMEDIATE:
        save_tif(LOD_DIR / f'LOD95_{key}.tif', lod95_grid, meta, 'float32', nodata=np.nan)
        save_tif(LOD_DIR / f'LOD95_sig_{key}.tif', signif, meta, 'uint8', nodata=0)
        save_tif(ZONE_DIR / f'ZONES_{key}.tif', zones_arr, z_meta, 'uint8', nodata=0)
        sigma_grid = lod95_grid / 1.96
        save_tif(UNC_DIR / f'sigma_diff_{key}.tif', sigma_grid, meta, 'float32', nodata=np.nan)
    df_vol = volume_table(diff, zones_arr, study_mask_bool)
    df_vol.to_csv(STAT_DIR / f'Vol_{key}.csv', index=False)
    vol_total = float(df_vol.loc[df_vol['Type'] == 'Total', 'Volume_m3'].values[0])
    log_kv('NetVolume_m3', vol_total)
    sigmaV = volume_uncertainty_sigmaV(signif, lod95_grid, is_grid, study_mask_bool)
    pd.DataFrame([{'Pair': key, 'LOD95_const_m': lod95_const_val, 'LOD95_mode': 'grid' if is_grid else 'const', 'SigmaV_m3': sigmaV}]).to_csv(STAT_DIR / f'VolumeUncertainty_{key}.csv', index=False)
    log_kv('SigmaV_m3', sigmaV)
    if MAKE_PREVIEW_PNG:
        plot_diff_preview(diff, key)
        vmax_lod = safe_nanpercentile(lod95_grid, 99, default=max(lod95_const_val, 1e-06))
        plot_array(np.where(np.isfinite(lod95_grid), lod95_grid, np.nan), PNG_DIR / f'LOD95_{key}.png', title=f'LOD95 {key}', cmap=cmap_from_hex(CMAP_DIFF_HEX), vmin=0, vmax=vmax_lod, label='LOD95 (m)')
        plot_zone_preview(zones_arr, key)
    transform = meta['transform']
    cen_E = centroid_from_zone_array(zones_arr, transform, 1)
    cen_D = centroid_from_zone_array(zones_arr, transform, 3)
    log_kv('Centroid_E', cen_E)
    log_kv('Centroid_D', cen_D)
    if MAKE_ZONAL_STATS and ZONAL_SHP:
        try:
            zonal_out_csv = STAT_DIR / f'Zonal_{key}.csv'
            zonal_stats_by_shp(diff, zones_arr, ZONAL_SHP, zonal_out_csv, study_mask_bool)
            log_kv('ZonalCSV', zonal_out_csv)
        except Exception as e:
            log(f'[WARN] zonal stats failed for {key}: {e}')
    return {'pair': key, 'later': later, 'first': first, 'pair_type': 'adjacent' if is_adjacent_pair(later, first) else 'cross', 'pixel_size': px, 'LOD95_const': lod95_const_val, 'is_grid': is_grid, 'Vol_total': vol_total, 'SigmaV': sigmaV, 'Centroid_E': cen_E, 'Centroid_D': cen_D, 'zones_tif': str(ZONE_DIR / f'ZONES_{key}.tif')}

def centroid_motion_metrics(c1: Optional[Tuple[float, float]], c2: Optional[Tuple[float, float]], dt: float) -> Optional[Dict[str, float]]:
    if c1 is None or c2 is None:
        return None
    dx = float(c2[0] - c1[0])
    dy = float(c2[1] - c1[1])
    dist = float(math.hypot(dx, dy))
    speed = float(dist / dt) if dt > 0 else np.nan
    angle = float(math.degrees(math.atan2(dy, dx)))
    return {'dx': dx, 'dy': dy, 'dist': dist, 'speed': speed, 'angle': angle}

def chain_analysis_adjacent(results: List[Dict[str, Union[str, float, bool, Optional[Tuple[float, float]]]]]) -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not results:
        log('[WARN] no adjacent results for chain analysis')
        return
    ordered: List[Dict[str, Union[str, float, bool, Optional[Tuple[float, float]]]]] = []
    pair_map = {(r['later'], r['first']): r for r in results}
    for p in ADJACENT_CHAIN_PAIRS:
        if p in pair_map:
            ordered.append(pair_map[p])
    if not ordered:
        log('[WARN] adjacent chain pairs not found')
        return
    if MAKE_SERIES_PLOT:
        plot_adjacent_series(ordered)
    rows = []
    for i in range(len(ordered) - 1):
        r1 = ordered[i]
        r2 = ordered[i + 1]
        me = centroid_motion_metrics(r1.get('Centroid_E'), r2.get('Centroid_E'), dt=1.0)
        md = centroid_motion_metrics(r1.get('Centroid_D'), r2.get('Centroid_D'), dt=1.0)
        row = {'from_pair': r1['pair'], 'to_pair': r2['pair'], 'pair_step': f"{r1['pair']} to {r2['pair']}"}
        if me is not None:
            row.update({'E_dx': me['dx'], 'E_dy': me['dy'], 'E_dist': me['dist'], 'E_speed': me['speed'], 'E_angle': me['angle']})
        else:
            row.update({'E_dx': None, 'E_dy': None, 'E_dist': None, 'E_speed': None, 'E_angle': None})
        if md is not None:
            row.update({'D_dx': md['dx'], 'D_dy': md['dy'], 'D_dist': md['dist'], 'D_speed': md['speed'], 'D_angle': md['angle']})
        else:
            row.update({'D_dx': None, 'D_dy': None, 'D_dist': None, 'D_speed': None, 'D_angle': None})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(STAT_DIR / 'Centroid_motion_adjacent.csv', index=False)
    log_kv('Centroid_motion_adjacent', STAT_DIR / 'Centroid_motion_adjacent.csv')
    if MAKE_SERIES_PLOT:
        plot_motion_vectors(rows, kind='E', fname='Erosion_motion_adjacent.png', color='blue')
        plot_motion_vectors(rows, kind='D', fname='Deposition_motion_adjacent.png', color='red')

def zonal_stats_by_shp(diff_path: Path, zones_arr: np.ndarray, shp_path: str, out_csv: Path, study_mask_bool: Optional[np.ndarray]) -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    import fiona
    from shapely.geometry import shape
    with rasterio.open(diff_path) as src:
        dz_ma = src.read(1, masked=True)
        dz = np.ma.filled(dz_ma, np.nan)
        transform = src.transform
        out_shape = dz.shape
    rows: List[Dict[str, Union[str, float]]] = []
    with fiona.open(shp_path) as shp:
        for ft in shp:
            geom = shape(ft['geometry'])
            name = ft['properties'].get(ZONAL_NAME_FIELD, ft['id'])
            poly_mask = geometry_mask([geom.__geo_interface__], transform=transform, invert=True, out_shape=out_shape, all_touched=False)
            valid = np.isfinite(dz) & poly_mask
            if study_mask_bool is not None:
                valid &= study_mask_bool

            def _one(code: int, nm: str) -> Dict[str, Union[str, float]]:
                m = valid & (zones_arr == code)
                area = float(m.sum() * CELL_AREA)
                if not m.any():
                    return dict(Zone=name, Type=nm, Area_m2=0.0, Mean_dz_m=0.0, Volume_m3=0.0)
                return dict(Zone=name, Type=nm, Area_m2=area, Mean_dz_m=float(np.nanmean(dz[m])), Volume_m3=float(np.nansum(dz[m]) * CELL_AREA))
            recs = [_one(1, 'Erosion'), _one(2, 'Stable'), _one(3, 'Deposition')]
            total_mask = valid
            total_area = float(total_mask.sum() * CELL_AREA)
            if total_mask.any():
                recs.append(dict(Zone=name, Type='Total', Area_m2=total_area, Mean_dz_m=float(np.nanmean(dz[total_mask])), Volume_m3=float(np.nansum(dz[total_mask]) * CELL_AREA)))
            else:
                recs.append(dict(Zone=name, Type='Total', Area_m2=0.0, Mean_dz_m=0.0, Volume_m3=0.0))
            rows.extend(recs)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def main() -> None:
    if LOG_TO_FILE and RUNLOG_PATH.exists():
        with open(RUNLOG_PATH, 'a', encoding='utf-8') as f:
            f.write('\n')
    log_section('Public-release status message.')
    log_kv('OUT_DIR', OUT_DIR)
    log_kv('RES(m)', RES)
    log_kv('AUTO_DISCOVER_PAIRS', AUTO_DISCOVER_PAIRS)
    log_kv('ALLOW_SKIP_MISSING', ALLOW_SKIP_MISSING)
    log_kv('PREFER_RES', PREFER_RES)
    log_kv('USE_SIGMA_RASTER', USE_SIGMA_RASTER)
    log_kv('STUDY_MASK_TIF', STUDY_MASK_TIF or 'None')
    log_kv('ZONAL_SHP', ZONAL_SHP or 'None')
    log_kv('PAIRS', PAIRS)
    log_kv('ADJACENT_CHAIN_PAIRS', ADJACENT_CHAIN_PAIRS)
    all_rows: List[Dict[str, Union[str, float, bool, Optional[Tuple[float, float]]]]] = []
    for (later, first) in PAIRS:
        result = run_one_pair(later, first)
        if not result:
            log(f'[INFO] skipped pair {later}-{first}')
            continue
        all_rows.append(result)
        log(f"[DONE] {result['pair']} | NetVol={result['Vol_total']:.2f} m3 | SigmaV={result['SigmaV']:.2f} m3")
    if not all_rows:
        log('Public-release status message.')
        log_section('Public-release status message.')
        return
    all_rows = sorted(all_rows, key=lambda r: pair_sort_key((str(r['later']), str(r['first']))))
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(STAT_DIR / 'Summary_All.csv', index=False)
    report_cols = ['pair', 'pair_type', 'later', 'first', 'pixel_size', 'LOD95_const', 'is_grid', 'Vol_total', 'SigmaV', 'Centroid_E', 'Centroid_D']
    use_cols = [c for c in report_cols if c in df_all.columns]
    df_all[use_cols].to_csv(RUNREP_PATH, index=False)
    log_kv('Summary_All', STAT_DIR / 'Summary_All.csv')
    log_kv('Run_Report', RUNREP_PATH)
    if MAKE_SERIES_PLOT:
        plot_pair_bar(all_rows)
    adjacent_rows = [r for r in all_rows if str(r.get('pair_type')) == 'adjacent']
    chain_analysis_adjacent(adjacent_rows)
    log_section('Public-release status message.')
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        err = f'Public-release status message.{e}'
        print(err)
        traceback.print_exc()
        if LOG_TO_FILE:
            with open(RUNLOG_PATH, 'a', encoding='utf-8') as f:
                f.write(_ts() + ' ' + err + '\n')
        sys.exit(1)
