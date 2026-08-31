"""Public-release documentation. Scientific logic and parameters are unchanged."""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.windows import Window
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / 'outputs'
BH_GNSS_CSV = PROJECT_ROOT / 'data' / 'site_catalog.csv'
SITE_ID_COL = 'ID'
SITE_NAME_COL = 'Num'
LON_COL = 'Longitude'
LAT_COL = 'Latitude'
REMARK_COL = '\u5907\u6ce8'
DIFF_DIR = PROJECT_ROOT / 'data' / 'dod_0p1m'
LOD_DIR = OUT_DIR / 'LOD95'
SCM_PROB_DIR = OUT_DIR / 'SCM_PROB'
SCM_BIN_DIR = OUT_DIR / 'SCM_BIN'
PATCH_VEC_DIR = OUT_DIR / 'PATCH' / 'VECTOR'
PATCH_FILE_PREFIX = 'PATCH_'
PAIRS_TO_USE: Optional[List[str]] = None
NEIGHBOR_RADIUS_M: Optional[float] = 2.0
USE_PATCH_DISTANCE: bool = True
GEOMORPH_OUT_DIR = OUT_DIR / 'PATCH'
GEOMORPH_OUT_CSV = GEOMORPH_OUT_DIR / 'BH_GNSS_geomorph_features.csv'
VERBOSE: bool = True
MAX_POINTS_PRINT: int = 5

def log(msg: str) -> None:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    print(msg)

def load_monitor_points(csv_path: Path) -> gpd.GeoDataFrame:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not csv_path.exists():
        raise FileNotFoundError(f'Public-release status message.{csv_path}')
    df = pd.read_csv(csv_path, encoding='gbk')
    log(f'Public-release status message.{csv_path}Public-release status message.{len(df)}')
    for col in [SITE_ID_COL, LON_COL, LAT_COL]:
        if col not in df.columns:
            raise KeyError(f'Public-release status message.{col}Public-release status message.{list(df.columns)}')
    if REMARK_COL is not None and REMARK_COL not in df.columns:
        df[REMARK_COL] = ''
    geometry = [Point(xy) for xy in zip(df[LON_COL].values, df[LAT_COL].values)]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs='EPSG:4326')
    log(f'Public-release status message.{gdf.crs}')
    return gdf

def discover_pairs_from_diff(diff_dir: Path, pairs_to_use: Optional[List[str]]=None) -> Dict[str, Dict[str, Path]]:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    pair_rasters: Dict[str, Dict[str, Path]] = {}
    if not diff_dir.exists():
        raise FileNotFoundError(f'Public-release status message.{diff_dir}')
    for fp in diff_dir.glob('*.tif'):
        stem = fp.stem
        legacy = re.search(r'(\d{6}-\d{6})', stem)
        release = re.search(r'(\d{8})_minus_(\d{8})', stem)
        if legacy:
            pair_str = legacy.group(1)
        elif release:
            pair_str = f'{release.group(1)[2:]}-{release.group(2)[2:]}'
        else:
            continue
        pair_rasters[pair_str] = {'diff': fp}
    if not pair_rasters:
        raise FileNotFoundError(f'Public-release status message.{diff_dir}Public-release status message.')
    if pairs_to_use is not None:
        pair_rasters = {p: pair_rasters[p] for p in pairs_to_use if p in pair_rasters}
        if not pair_rasters:
            raise ValueError(f'Public-release status message.')
    for pair_str in pair_rasters.keys():
        lod_fp = LOD_DIR / f'LOD95_sig_{pair_str}.tif'
        prob_fp = SCM_PROB_DIR / f'SCM_prob_{pair_str}.tif'
        bin_fp = SCM_BIN_DIR / f'SCM_bin_{pair_str}_th0.50.tif'
        if lod_fp.exists():
            pair_rasters[pair_str]['lod'] = lod_fp
        if prob_fp.exists():
            pair_rasters[pair_str]['scm_prob'] = prob_fp
        if bin_fp.exists():
            pair_rasters[pair_str]['scm_bin'] = bin_fp
    log(f'Public-release status message.{diff_dir}Public-release status message.{len(pair_rasters)}Public-release status message.')
    for (pair_str, paths) in pair_rasters.items():
        log(f"    pair={pair_str} -> DIFF={paths['diff'].name}")
    return pair_rasters

def sample_raster_at_points(raster_path: Path, gdf_points_proj: gpd.GeoDataFrame, nodata_value: Optional[float]=None) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not raster_path.exists():
        raise FileNotFoundError(f'Public-release status message.{raster_path}')
    n_pts = len(gdf_points_proj)
    values = np.full(n_pts, np.nan, dtype=float)
    with rasterio.open(raster_path) as ds:
        if nodata_value is None:
            nodata_value = ds.nodata
        xs = gdf_points_proj.geometry.x.values
        ys = gdf_points_proj.geometry.y.values
        mask_valid = np.isfinite(xs) & np.isfinite(ys)
        n_valid = int(mask_valid.sum())
        n_invalid = n_pts - n_valid
        if n_invalid > 0:
            log(f'Public-release status message.{n_invalid}Public-release status message.')
        if n_valid == 0:
            return values
        xs_valid = xs[mask_valid]
        ys_valid = ys[mask_valid]
        coords_valid = list(zip(xs_valid, ys_valid))
        sampled = list(ds.sample(coords_valid))
        arr_valid = np.array(sampled).reshape(-1)
        if nodata_value is not None:
            arr_valid = np.where(arr_valid == nodata_value, np.nan, arr_valid)
        values[mask_valid] = arr_valid
    return values

def sample_raster_neighborhood_mean(raster_path: Path, gdf_points_proj: gpd.GeoDataFrame, radius_m: float) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    n_pts = len(gdf_points_proj)
    mean_vals = np.full(n_pts, np.nan, dtype=float)
    if radius_m is None or radius_m <= 0:
        return mean_vals
    if not raster_path.exists():
        raise FileNotFoundError(f'Public-release status message.{raster_path}')
    with rasterio.open(raster_path) as ds:
        res_x = ds.transform.a
        pixel_size = abs(res_x)
        if pixel_size <= 0:
            raise ValueError(f'Public-release status message.{raster_path}Public-release status message.{pixel_size}')
        radius_px = int(round(radius_m / pixel_size))
        if radius_px < 1:
            radius_px = 1
        width = ds.width
        height = ds.height
        nodata_value = ds.nodata
        xs = gdf_points_proj.geometry.x.values
        ys = gdf_points_proj.geometry.y.values
        n_invalid_coord = int((~np.isfinite(xs) | ~np.isfinite(ys)).sum())
        if n_invalid_coord > 0:
            log(f'Public-release status message.{n_invalid_coord}Public-release status message.')
        for (i, geom) in enumerate(gdf_points_proj.geometry):
            (x, y) = (geom.x, geom.y)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            try:
                (row, col) = ds.index(x, y)
            except Exception:
                continue
            if row < 0 or row >= height or col < 0 or (col >= width):
                continue
            row_start = max(row - radius_px, 0)
            row_stop = min(row + radius_px + 1, height)
            col_start = max(col - radius_px, 0)
            col_stop = min(col + radius_px + 1, width)
            win = Window.from_slices((row_start, row_stop), (col_start, col_stop))
            data = ds.read(1, window=win).astype(float)
            if nodata_value is not None:
                data[data == nodata_value] = np.nan
            if np.all(np.isnan(data)):
                continue
            mean_vals[i] = float(np.nanmean(data))
    return mean_vals

def compute_distance_to_patch_union(patch_fp: Path, gdf_points_proj: gpd.GeoDataFrame) -> np.ndarray:
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    if not patch_fp.exists():
        return np.full(len(gdf_points_proj), np.nan, dtype=float)
    gdf_patch = gpd.read_file(patch_fp)
    if gdf_patch.empty:
        return np.full(len(gdf_points_proj), np.nan, dtype=float)
    if gdf_patch.crs is None:
        raise ValueError(f'Public-release status message.{patch_fp}Public-release status message.')
    if gdf_points_proj.crs is None:
        raise ValueError('Public-release status message.')
    if str(gdf_points_proj.crs) != str(gdf_patch.crs):
        gdf_points_proj = gdf_points_proj.to_crs(gdf_patch.crs)
    union_geom = gdf_patch.unary_union
    if union_geom.is_empty:
        return np.full(len(gdf_points_proj), np.nan, dtype=float)
    distances = []
    for geom in gdf_points_proj.geometry:
        distances.append(float(geom.distance(union_geom)))
    return np.array(distances, dtype=float)

def main():
    """Public-release documentation. Scientific logic and parameters are unchanged."""
    t_start = time.time()
    log('======================================================================')
    log('Public-release status message.')
    log('======================================================================')
    GEOMORPH_OUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf_points_ll = load_monitor_points(BH_GNSS_CSV)
    pair_rasters = discover_pairs_from_diff(DIFF_DIR, PAIRS_TO_USE)
    records: List[pd.DataFrame] = []
    for (pair_str, rasters) in pair_rasters.items():
        log('\n----------------------------------------------------------------------')
        log(f'Public-release status message.{pair_str}')
        log('----------------------------------------------------------------------')
        diff_fp = rasters['diff']
        lod_fp = rasters.get('lod', None)
        prob_fp = rasters.get('scm_prob', None)
        bin_fp = rasters.get('scm_bin', None)
        log(f'Public-release status message.{diff_fp}')
        if lod_fp is not None:
            log(f'Public-release status message.{lod_fp}')
        else:
            log('Public-release status message.')
        if prob_fp is not None:
            log(f'Public-release status message.{prob_fp}')
        else:
            log('Public-release status message.')
        if bin_fp is not None:
            log(f'Public-release status message.{bin_fp}')
        else:
            log('Public-release status message.')
        with rasterio.open(diff_fp) as ds_diff:
            diff_crs = ds_diff.crs
        if diff_crs is None:
            raise ValueError(f'Public-release status message.{diff_fp}Public-release status message.')
        gdf_points_proj = gdf_points_ll.to_crs(diff_crs)
        log(f'Public-release status message.{gdf_points_ll.crs}Public-release status message.{diff_crs}')
        dz_point = sample_raster_at_points(diff_fp, gdf_points_proj, nodata_value=None)
        if NEIGHBOR_RADIUS_M is not None and NEIGHBOR_RADIUS_M > 0:
            log(f'Public-release status message.{NEIGHBOR_RADIUS_M} m ...')
            dz_mean_local = sample_raster_neighborhood_mean(diff_fp, gdf_points_proj, radius_m=NEIGHBOR_RADIUS_M)
        else:
            dz_mean_local = np.full(len(gdf_points_proj), np.nan, dtype=float)
        if lod_fp is not None:
            lod_vals = sample_raster_at_points(lod_fp, gdf_points_proj, nodata_value=None)
        else:
            lod_vals = np.full(len(gdf_points_proj), np.nan, dtype=float)
        if prob_fp is not None:
            scm_prob_vals = sample_raster_at_points(prob_fp, gdf_points_proj, nodata_value=None)
        else:
            scm_prob_vals = np.full(len(gdf_points_proj), np.nan, dtype=float)
        if bin_fp is not None:
            scm_bin_vals = sample_raster_at_points(bin_fp, gdf_points_proj, nodata_value=None)
        else:
            scm_bin_vals = np.full(len(gdf_points_proj), np.nan, dtype=float)
        if USE_PATCH_DISTANCE:
            patch_fp = PATCH_VEC_DIR / f'{PATCH_FILE_PREFIX}{pair_str}.gpkg'
            log(f'Public-release status message.{patch_fp}')
            dist_vals = compute_distance_to_patch_union(patch_fp, gdf_points_proj)
        else:
            dist_vals = np.full(len(gdf_points_proj), np.nan, dtype=float)
        df_pair = pd.DataFrame({'site_id': gdf_points_ll[SITE_ID_COL].values, 'site_name': gdf_points_ll[SITE_NAME_COL].values, 'lon': gdf_points_ll[LON_COL].values, 'lat': gdf_points_ll[LAT_COL].values, 'remark': gdf_points_ll[REMARK_COL].values if REMARK_COL is not None and REMARK_COL in gdf_points_ll.columns else '', 'x': gdf_points_proj.geometry.x.values, 'y': gdf_points_proj.geometry.y.values, 'pair': pair_str, 'dz_point': dz_point, 'abs_dz_point': np.abs(dz_point), 'dz_mean_local': dz_mean_local, 'LOD_sig': lod_vals, 'SCM_prob': scm_prob_vals, 'SCM_bin': scm_bin_vals, 'dist_to_patch_m': dist_vals})
        records.append(df_pair)
        if VERBOSE:
            log(f'[PAIR] {pair_str}Public-release status message.{len(df_pair)}Public-release status message.{MAX_POINTS_PRINT}Public-release status message.')
            log(df_pair.head(MAX_POINTS_PRINT).to_string(index=False))
    df_all = pd.concat(records, axis=0, ignore_index=True)
    GEOMORPH_OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(GEOMORPH_OUT_CSV, index=False, encoding='utf-8-sig')
    t_end = time.time()
    log('\n======================================================================')
    log(f'Public-release status message.{GEOMORPH_OUT_CSV}')
    log(f'Public-release status message.{t_end - t_start:.2f}Public-release status message.')
    log('======================================================================')
if __name__ == '__main__':
    main()
