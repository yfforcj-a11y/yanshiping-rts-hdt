"""Public-release documentation. Scientific logic and parameters are unchanged."""
import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
import rasterio.windows as rio_windows
try:
    from skimage import morphology, measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
try:
    import geopandas as gpd
    from shapely.geometry import shape as shp_shape
    HAS_GPD = True
except ImportError:
    HAS_GPD = False
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
OUT_DIR = Path(__file__).resolve().parents[2] / 'outputs'
DIFF_DIR = Path(__file__).resolve().parents[2] / 'data' / 'dod_0p1m'
LOD95_DIR = OUT_DIR / 'LOD95'
SCM_BIN_DIR = OUT_DIR / 'SCM_BIN'
SCM_PROB_DIR = OUT_DIR / 'SCM_PROB'
PATCH_ROOT_DIR = OUT_DIR / 'PATCH'
PATCH_RAS_DIR = PATCH_ROOT_DIR / 'RASTER'
PATCH_VEC_DIR = PATCH_ROOT_DIR / 'VECTOR'
PATCH_STAT_DIR = PATCH_ROOT_DIR / 'STATS'
PATCH_PNG_DIR = PATCH_ROOT_DIR / 'PNG'
for d in [PATCH_ROOT_DIR, PATCH_RAS_DIR, PATCH_VEC_DIR, PATCH_STAT_DIR, PATCH_PNG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
TAGS: List[str] = ['230924', '240630', '250816', '251017']
FIRST_TAG: str = TAGS[0]
DEM_RES: float = 0.1
CELL_AREA: float = DEM_RES * DEM_RES
USE_LOD95_MASK: bool = True
MIN_ABS_DZ: float = 0.5
SIGN_MODE: str = 'erosion_only'
SCM_BIN_VALUE_FOR_CHANGE: int = 1
STUDY_MASK_TIF: Optional[str] = None
MASK_AS_BOOL: bool = True
USE_MORPH_FILTER: bool = True
MORPH_OPEN_SIZE: int = 3
MORPH_CLOSE_SIZE: int = 3
MIN_PATCH_PIXELS: int = 50
MIN_PATCH_AREA_M2: float = 0.0
WRITE_PATCH_RASTER: bool = True
WRITE_PATCH_STATS_CSV: bool = True
WRITE_PATCH_VECTOR: bool = True
VECTOR_FORMAT: str = 'gpkg'
MAKE_PREVIEW_PNG: bool = False
PNG_DPI: int = 300
DIFF_SYM_RANGE: float = 12.0
LOG_LEVEL = logging.INFO
MAX_PAIRS_TO_PROCESS: Optional[int] = None

def setup_logger() -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s [%(levelname)s] %(message)s')

def discover_pairs_from_scm_bin() -> List[str]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    pairs: List[str] = []
    if not SCM_BIN_DIR.exists():
        logging.error(f'Public-release status message.{SCM_BIN_DIR}')
        return pairs
    for tif_path in SCM_BIN_DIR.glob('SCM_bin_*_th*.tif'):
        name = tif_path.name
        try:
            core = name.replace('SCM_bin_', '')
            core = core.split('_th')[0]
            if '-' in core:
                pairs.append(core)
        except Exception:
            continue
    pairs = sorted(set(pairs))
    logging.info(f'Public-release status message.{len(pairs)}Public-release status message.{pairs}')
    return pairs

def build_path_for_pair(pair: str) -> Dict[str, Optional[Path]]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    (later, first) = pair.split('-')
    diff_path = DIFF_DIR / f'yanshiping_rts_dod_20{later}_minus_20{first}_0p1m.tif'
    lod_path = LOD95_DIR / f'LOD95_sig_{later}-{first}.tif'
    scm_bin_path = SCM_BIN_DIR / f'SCM_bin_{pair}_th0.50.tif'
    scm_prob_path = SCM_PROB_DIR / f'SCM_prob_{pair}.tif'
    patch_raster_path = PATCH_RAS_DIR / f'PATCH_ID_{pair}.tif'
    patch_stats_path = PATCH_STAT_DIR / f'PATCH_stats_{pair}.csv'
    if VECTOR_FORMAT.lower() == 'gpkg':
        patch_vector_path = PATCH_VEC_DIR / f'PATCH_{pair}.gpkg'
    elif VECTOR_FORMAT.lower() == 'shp':
        patch_vector_path = PATCH_VEC_DIR / f'PATCH_{pair}.shp'
    else:
        patch_vector_path = None
    preview_png_path = PATCH_PNG_DIR / f'PATCH_preview_{pair}.png'
    paths = dict(diff=diff_path, lod=lod_path, scm_bin=scm_bin_path, scm_prob=scm_prob_path, patch_raster=patch_raster_path, patch_stats=patch_stats_path, patch_vector=patch_vector_path, preview_png=preview_png_path)
    return paths

def read_single_band(path: Path) -> Tuple[np.ndarray, Affine, dict]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not path.exists():
        raise FileNotFoundError(f'Public-release status message.{path}')
    with rasterio.open(path) as ds:
        data = ds.read(1)
        transform = ds.transform
        meta = ds.meta.copy()
    return (data, transform, meta)

def apply_study_mask(data: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if mask is None:
        return data
    return data

def build_final_change_mask(diff: np.ndarray, scm_bin: np.ndarray, lod95: Optional[np.ndarray], study_mask: Optional[np.ndarray], nodata_diff: Optional[float]) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    valid = np.ones_like(diff, dtype=bool)
    if nodata_diff is not None:
        valid &= diff != nodata_diff
    valid &= np.isfinite(diff)
    if study_mask is not None:
        valid &= study_mask
    scm_change = scm_bin == SCM_BIN_VALUE_FOR_CHANGE
    if USE_LOD95_MASK and lod95 is not None:
        lod_ok = lod95 == 1
    else:
        lod_ok = np.ones_like(diff, dtype=bool)
    if MIN_ABS_DZ > 0:
        dz_ok = np.abs(diff) >= MIN_ABS_DZ
    else:
        dz_ok = np.ones_like(diff, dtype=bool)
    if SIGN_MODE == 'erosion_only':
        sign_ok = diff < 0.0
    elif SIGN_MODE == 'deposition_only':
        sign_ok = diff > 0.0
    else:
        sign_ok = np.ones_like(diff, dtype=bool)
    final_mask = valid & scm_change & lod_ok & dz_ok & sign_ok
    return final_mask

def morph_filter(mask: np.ndarray) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not HAS_SKIMAGE:
        logging.warning('Public-release status message.')
        return mask
    filtered = mask.copy()
    if MORPH_OPEN_SIZE > 1:
        selem_open = morphology.square(MORPH_OPEN_SIZE)
        filtered = morphology.opening(filtered, selem_open)
    if MORPH_CLOSE_SIZE > 1:
        selem_close = morphology.square(MORPH_CLOSE_SIZE)
        filtered = morphology.closing(filtered, selem_close)
    return filtered

def remove_small_patches(mask: np.ndarray, min_pixels: int, min_area_m2: float) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not HAS_SKIMAGE:
        logging.warning('Public-release status message.')
        return mask
    (labeled, num) = measure.label(mask, connectivity=2, return_num=True)
    logging.info(f'Public-release status message.{num}')
    if num == 0:
        return mask
    filtered = np.zeros_like(mask, dtype=bool)
    for label_id in range(1, num + 1):
        region_mask = labeled == label_id
        pix_count = int(region_mask.sum())
        area_m2 = pix_count * CELL_AREA
        if pix_count >= min_pixels and area_m2 >= min_area_m2:
            filtered |= region_mask
    return filtered

def label_patches(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not HAS_SKIMAGE:
        logging.warning('Public-release status message.')
        labels = np.zeros_like(mask, dtype=np.int32)
        return (labels, 0)
    (labels, num) = measure.label(mask, connectivity=2, return_num=True)
    labels = labels.astype(np.int32)
    logging.info(f'Public-release status message.{num}')
    return (labels, num)

def compute_patch_stats(labels: np.ndarray, diff: np.ndarray, pair: str) -> pd.DataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    patch_ids = np.unique(labels)
    patch_ids = patch_ids[patch_ids > 0]
    records = []
    for pid in patch_ids:
        mask = labels == pid
        pix_count = int(mask.sum())
        if pix_count == 0:
            continue
        dz_vals = diff[mask]
        dz_vals = dz_vals[np.isfinite(dz_vals)]
        if dz_vals.size == 0:
            continue
        area_m2 = pix_count * CELL_AREA
        dz_mean = float(np.mean(dz_vals))
        dz_median = float(np.median(dz_vals))
        dz_min = float(np.min(dz_vals))
        dz_max = float(np.max(dz_vals))
        dz_std = float(np.std(dz_vals))
        volume_m3 = dz_mean * area_m2
        records.append(dict(pair=pair, patch_id=int(pid), pixel_count=pix_count, area_m2=area_m2, dz_mean=dz_mean, dz_median=dz_median, dz_min=dz_min, dz_max=dz_max, dz_std=dz_std, volume_m3=volume_m3))
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values(by='patch_id').reset_index(drop=True)
    return df

def write_raster_int32(out_path: Path, data: np.ndarray, transform: Affine, meta_template: dict, nodata_value: int=0) -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    meta = meta_template.copy()
    meta.update(driver='GTiff', dtype='int32', count=1, nodata=nodata_value, transform=transform, width=data.shape[1], height=data.shape[0], compress='LZW', tiled=True, bigtiff='IF_SAFER', blockxsize=256, blockysize=256)
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(data.astype(np.int32), 1)
    logging.info(f'Public-release status message.{out_path}')

def vectorize_patches(labels: np.ndarray, transform: Affine, crs: dict, stats_df: pd.DataFrame, out_path: Path) -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not HAS_GPD:
        logging.warning('Public-release status message.')
        return
    labels_int = labels.astype('int32')
    geoms = []
    vals = []
    for (geom, value) in shapes(labels_int, transform=transform):
        if value == 0:
            continue
        geoms.append(shp_shape(geom))
        vals.append(int(value))
    if not geoms:
        logging.warning('Public-release status message.')
        return
    gdf = gpd.GeoDataFrame({'patch_id': vals}, geometry=geoms, crs=crs)
    if stats_df is not None and (not stats_df.empty):
        gdf = gdf.merge(stats_df, on='patch_id', how='left')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == '.gpkg':
        gdf.to_file(out_path, layer='patches', driver='GPKG')
    elif out_path.suffix.lower() == '.shp':
        gdf.to_file(out_path)
    else:
        gdf.to_file(out_path.with_suffix('.gpkg'), layer='patches', driver='GPKG')
    logging.info(f'Public-release status message.{out_path}')

def make_preview_png(diff: np.ndarray, labels: np.ndarray, out_path: Path, title: str='') -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not (HAS_MPL and HAS_SKIMAGE):
        logging.warning('Public-release status message.')
        return
    (fig, ax) = plt.subplots(figsize=(8, 6), dpi=PNG_DPI)
    (vmin, vmax) = (-DIFF_SYM_RANGE, DIFF_SYM_RANGE)
    im = ax.imshow(diff, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Elevation change (m)')
    mask_patch = labels > 0
    contours = measure.find_contours(mask_patch.astype(float), level=0.5)
    for cnt in contours:
        ax.plot(cnt[:, 1], cnt[:, 0], color='black', linewidth=0.5)
    ax.set_title(title or 'Patch preview')
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PNG_DPI, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'Public-release status message.{out_path}')

def process_pair(pair: str) -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    logging.info(f'Public-release status message.{pair} =====')
    paths = build_path_for_pair(pair)
    diff_path = paths['diff']
    lod_path = paths['lod']
    scm_bin_path = paths['scm_bin']
    scm_prob_path = paths['scm_prob']
    patch_raster_path = paths['patch_raster']
    patch_stats_path = paths['patch_stats']
    patch_vector_path = paths['patch_vector']
    preview_png_path = paths['preview_png']
    with rasterio.open(scm_bin_path) as ds_scm:
        scm_bin = ds_scm.read(1)
        scm_transform = ds_scm.transform
        scm_height = ds_scm.height
        scm_width = ds_scm.width
        scm_crs = ds_scm.crs
        scm_bounds = ds_scm.bounds
    logging.info(f'Public-release status message.{scm_bin_path}Public-release status message.{scm_bin.shape}, bounds={scm_bounds}')
    with rasterio.open(diff_path) as ds_diff:
        logging.info(f'Public-release status message.{diff_path}')
        window_diff = rio_windows.from_bounds(left=scm_bounds.left, bottom=scm_bounds.bottom, right=scm_bounds.right, top=scm_bounds.top, transform=ds_diff.transform)
        window_diff = window_diff.round_offsets().round_lengths()
        diff = ds_diff.read(1, window=window_diff)
        diff_transform = ds_diff.window_transform(window_diff)
        diff_meta = ds_diff.meta.copy()
        diff_meta.update(width=diff.shape[1], height=diff.shape[0], transform=diff_transform)
        nodata_diff = diff_meta.get('nodata', ds_diff.nodata)
    logging.info(f'Public-release status message.{diff.shape}, transform={diff_transform}')
    if USE_LOD95_MASK:
        if lod_path.exists():
            with rasterio.open(lod_path) as ds_lod:
                lod95 = ds_lod.read(1, window=window_diff)
            logging.info(f'Public-release status message.{lod_path}Public-release status message.')
        else:
            logging.warning(f'Public-release status message.{lod_path}Public-release status message.')
            lod95 = None
    else:
        lod95 = None
    if STUDY_MASK_TIF is not None:
        mask_path = Path(STUDY_MASK_TIF)
        if mask_path.exists():
            with rasterio.open(mask_path) as ds_mask:
                study_mask_arr = ds_mask.read(1, window=window_diff)
            if MASK_AS_BOOL:
                study_mask = study_mask_arr != 0
            else:
                study_mask = study_mask_arr.astype(bool)
            logging.info(f'Public-release status message.{STUDY_MASK_TIF}Public-release status message.{study_mask.shape}')
        else:
            logging.warning(f'Public-release status message.{mask_path}Public-release status message.')
            study_mask = None
    else:
        study_mask = None
    if diff.shape != scm_bin.shape:
        logging.warning(f'Public-release status message.{diff.shape}Public-release status message.{scm_bin.shape}Public-release status message.')
        scm_bin = scm_bin[:diff.shape[0], :diff.shape[1]]
        if lod95 is not None and lod95.shape != diff.shape:
            lod95 = lod95[:diff.shape[0], :diff.shape[1]]
        if study_mask is not None and study_mask.shape != diff.shape:
            study_mask = study_mask[:diff.shape[0], :diff.shape[1]]
    base_mask = build_final_change_mask(diff=diff, scm_bin=scm_bin, lod95=lod95, study_mask=study_mask, nodata_diff=nodata_diff)
    logging.info(f'Public-release status message.{int(base_mask.sum())}')
    if base_mask.sum() == 0:
        logging.warning(f'Public-release status message.{pair}Public-release status message.')
        return
    if USE_MORPH_FILTER:
        base_mask = morph_filter(base_mask)
    if MIN_PATCH_PIXELS > 1 or MIN_PATCH_AREA_M2 > 0:
        base_mask = remove_small_patches(mask=base_mask, min_pixels=MIN_PATCH_PIXELS, min_area_m2=MIN_PATCH_AREA_M2)
    logging.info(f'Public-release status message.{int(base_mask.sum())}')
    if base_mask.sum() == 0:
        logging.warning(f'Public-release status message.{pair}Public-release status message.')
        return
    (labels, num_labels) = label_patches(base_mask)
    if num_labels == 0:
        logging.warning(f'Public-release status message.{pair}Public-release status message.')
        return
    stats_df = compute_patch_stats(labels=labels, diff=diff, pair=pair)
    if stats_df is None or stats_df.empty:
        logging.warning(f'Public-release status message.{pair}Public-release status message.')
        return
    if WRITE_PATCH_RASTER:
        write_raster_int32(out_path=patch_raster_path, data=labels, transform=diff_transform, meta_template=diff_meta, nodata_value=0)
    if WRITE_PATCH_STATS_CSV:
        patch_stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(patch_stats_path, index=False, encoding='utf-8-sig')
        logging.info(f'Public-release status message.{patch_stats_path}')
    if WRITE_PATCH_VECTOR and HAS_GPD:
        vectorize_patches(labels=labels, transform=diff_transform, crs=diff_meta.get('crs', scm_crs), stats_df=stats_df, out_path=patch_vector_path)
    elif WRITE_PATCH_VECTOR and (not HAS_GPD):
        logging.warning('Public-release status message.')
    if MAKE_PREVIEW_PNG:
        make_preview_png(diff=diff, labels=labels, out_path=preview_png_path, title=f'Patch preview {pair}')
    logging.info(f'Public-release status message.{pair}Public-release status message.')

def main():
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    setup_logger()
    logging.info('===== [7] RTS Patch Post-Processing & Vectorization Started =====')
    pairs = discover_pairs_from_scm_bin()
    if not pairs:
        logging.error('Public-release status message.')
        return
    if MAX_PAIRS_TO_PROCESS is not None:
        pairs = pairs[:MAX_PAIRS_TO_PROCESS]
    for pair in pairs:
        try:
            process_pair(pair)
        except Exception as e:
            logging.exception(f'Public-release status message.{pair}Public-release status message.{e}')
    logging.info('Public-release status message.')
if __name__ == '__main__':
    main()
