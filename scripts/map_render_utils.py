from __future__ import annotations
import json
import math
import os
import re
import struct
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
import numpy as np
try:
    import rasterio
except ModuleNotFoundError:
    rasterio = None
try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None
try:
    from pyproj import Transformer
except ModuleNotFoundError:
    Transformer = None
try:
    import laspy
except ModuleNotFoundError:
    laspy = None
from utils_project_paths import assert_not_deprecated, get_data_source, list_latest_files, load_manifest
MORANDI_INK = '#2F3437'
NODATA_COLOR = '#F4F3EF'
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

@dataclass(frozen=True)
class Bounds:
    left: float
    bottom: float
    right: float
    top: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bottom + self.top) / 2.0

    def to_dict(self) -> dict[str, float]:
        return {'left': float(self.left), 'bottom': float(self.bottom), 'right': float(self.right), 'top': float(self.top), 'width': float(self.width), 'height': float(self.height)}

@dataclass
class RasterLayer:
    path: Path
    label: str
    array: np.ndarray
    bounds: Bounds
    valid_extent: Bounds
    crs: str
    res_x: float
    res_y: float
    width: int
    height: int
    nodata: float | None

def parse_yyyymmdd(token: str) -> date:
    yy = int(token[:2])
    return date(2000 + yy if yy < 80 else 1900 + yy, int(token[2:4]), int(token[4:6]))

def date_label(token: str) -> str:
    return parse_yyyymmdd(token).strftime('%Y-%m-%d')

def parse_epoch_from_name(path: Path) -> str:
    match = re.search('(?<!\\d)(\\d{6})(?!\\d)', path.stem)
    if not match:
        raise ValueError(f'Could not parse epoch date from filename: {path.name}')
    return match.group(1)

def parse_pair_from_name(path: Path) -> str:
    match = re.search('(?<!\\d)(\\d{6}-\\d{6})(?!\\d)', path.stem)
    if match:
        return match.group(1)
    release = re.search('(?<!\\d)(\\d{8})_minus_(\\d{8})(?!\\d)', path.stem)
    if release:
        return f'{release.group(1)[2:]}-{release.group(2)[2:]}'
    raise ValueError(f'Could not parse DoD pair from filename: {path.name}')

def pair_label(pair: str) -> str:
    (later, earlier) = pair.split('-')
    return f'{date_label(later)} - {date_label(earlier)}'

def sort_dem_files(files: list[Path]) -> list[Path]:
    return sorted(files, key=lambda p: parse_epoch_from_name(p))

def sort_dod_files(files: list[Path]) -> list[Path]:
    preferred = ['240630-230924', '250816-240630', '251017-250816', '250816-230924', '251017-240630', '251017-230924']
    rank = {pair: idx for (idx, pair) in enumerate(preferred)}
    return sorted(files, key=lambda p: (rank.get(parse_pair_from_name(p), 999), parse_pair_from_name(p)))

def selected_manifest_files(source_key: str, pattern_key: str, expected_count: int, sort_kind: str, manifest: dict[str, Any] | None=None) -> list[Path]:
    manifest = manifest or load_manifest()
    root = get_data_source(source_key, manifest=manifest)
    assert_not_deprecated(root, manifest=manifest)
    files = list_latest_files(source_key, pattern_key, manifest=manifest)
    if sort_kind == 'dem':
        files = sort_dem_files(files)
    elif sort_kind == 'dod':
        files = sort_dod_files(files)
    for path in files:
        assert_not_deprecated(path, manifest=manifest)
        if not is_under(path, root):
            raise ValueError(f'Selected file is outside manifest root {source_key}: {path}')
    if len(files) != expected_count:
        raise RuntimeError(f'{source_key}/{pattern_key} expected {expected_count} files, found {len(files)}: {[p.name for p in files]}')
    return files

def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False

def _pil_nodata(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='ignore')
    try:
        return float(str(value).strip())
    except ValueError:
        return None

def _epsg_from_geokeys(value: object) -> str:
    if not value:
        return ''
    vals = list(value)
    for idx in range(4, len(vals) - 3, 4):
        (key_id, tag_location, count, val_offset) = vals[idx:idx + 4]
        if key_id == 3072 and tag_location == 0 and (count == 1):
            return f'EPSG:{val_offset}'
    return ''

def read_raster_layer(path: Path, label: str) -> RasterLayer:
    assert_not_deprecated(path)
    if rasterio is not None:
        with rasterio.open(path) as src:
            array = src.read(1, masked=True).astype('float32').filled(np.nan)
            nodata = float(src.nodata) if src.nodata is not None else None
            if nodata is not None:
                array[np.isclose(array, nodata, rtol=0.0, atol=1e-06)] = np.nan
            bounds = Bounds(float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top))
            (res_x, res_y) = (float(abs(src.res[0])), float(abs(src.res[1])))
            crs = str(src.crs) if src.crs is not None else ''
            valid = valid_extent_from_array(array, bounds, res_x, res_y)
            return RasterLayer(path.resolve(), label, array, bounds, valid, crs, res_x, res_y, int(src.width), int(src.height), nodata)
    if Image is None:
        raise ModuleNotFoundError('Reading GeoTIFF requires rasterio or PIL.')
    with Image.open(path) as im:
        tags = im.tag_v2
        (width, height) = im.size
        pixel_scale = tags.get(33550)
        tiepoint = tags.get(33922)
        if not pixel_scale or not tiepoint:
            raise ValueError(f'GeoTIFF lacks required georeference tags: {path}')
        (res_x, res_y) = (float(pixel_scale[0]), float(pixel_scale[1]))
        (i0, j0, _, x0, y0, _) = [float(v) for v in tiepoint[:6]]
        left = x0 - i0 * res_x
        top = y0 + j0 * res_y
        bounds = Bounds(left, top - height * res_y, left + width * res_x, top)
        nodata = _pil_nodata(tags.get(42113))
        crs = _epsg_from_geokeys(tags.get(34735))
        array = np.asarray(im, dtype='float32').copy()
    array[~np.isfinite(array)] = np.nan
    if nodata is not None:
        array[np.isclose(array, nodata, rtol=0.0, atol=1e-06)] = np.nan
    valid = valid_extent_from_array(array, bounds, res_x, res_y)
    return RasterLayer(path.resolve(), label, array, bounds, valid, crs, res_x, res_y, int(width), int(height), nodata)

def valid_extent_from_array(array: np.ndarray, bounds: Bounds, res_x: float, res_y: float) -> Bounds:
    mask = np.isfinite(array)
    if not mask.any():
        raise ValueError('Raster valid extent is empty.')
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    left = bounds.left + float(cols.min()) * res_x
    right = bounds.left + float(cols.max() + 1) * res_x
    top = bounds.top - float(rows.min()) * res_y
    bottom = bounds.top - float(rows.max() + 1) * res_y
    return Bounds(left, bottom, right, top)

def read_las_header_bounds(path: Path) -> Bounds:
    assert_not_deprecated(path)
    if laspy is not None:
        with laspy.open(path) as reader:
            header = reader.header
            return Bounds(float(header.mins[0]), float(header.mins[1]), float(header.maxs[0]), float(header.maxs[1]))
    with path.open('rb') as f:
        header = f.read(227)
    if len(header) < 227 or header[:4] != b'LASF':
        raise ValueError(f'Not a readable LAS file header: {path}')
    (max_x, min_x, max_y, min_y, _max_z, _min_z) = struct.unpack_from('<6d', header, 179)
    return Bounds(min_x, min_y, max_x, max_y)

def union_bounds(bounds_list: list[Bounds]) -> Bounds:
    if not bounds_list:
        raise ValueError('Cannot union an empty bounds list.')
    return Bounds(left=min((b.left for b in bounds_list)), bottom=min((b.bottom for b in bounds_list)), right=max((b.right for b in bounds_list)), top=max((b.top for b in bounds_list)))

def extent_deltas(a: Bounds, b: Bounds) -> dict[str, float]:
    return {'left_m': abs(a.left - b.left), 'bottom_m': abs(a.bottom - b.bottom), 'right_m': abs(a.right - b.right), 'top_m': abs(a.top - b.top), 'max_m': max(abs(a.left - b.left), abs(a.bottom - b.bottom), abs(a.right - b.right), abs(a.top - b.top))}

def validate_raster_las_extents(raster_layers: list[RasterLayer], las_files: list[Path], tolerance_m: float) -> list[dict[str, Any]]:
    if len(raster_layers) != len(las_files):
        raise RuntimeError(f'Raster/LAS count mismatch: {len(raster_layers)} rasters vs {len(las_files)} LAS files.')
    rows: list[dict[str, Any]] = []
    for (layer, las_path) in zip(raster_layers, las_files):
        las_bounds = read_las_header_bounds(las_path)
        deltas = extent_deltas(layer.valid_extent, las_bounds)
        ok = deltas['max_m'] <= tolerance_m
        row = {'label': layer.label, 'raster_path': str(layer.path), 'las_path': str(las_path.resolve()), 'raster_valid_extent': layer.valid_extent.to_dict(), 'las_header_extent': las_bounds.to_dict(), 'delta_m': deltas, 'within_tolerance': ok}
        rows.append(row)
        if not ok:
            raise RuntimeError(f"Raster/LAS footprint mismatch for {layer.label}: max delta {deltas['max_m']:.3f} m > {tolerance_m:.3f} m")
    return rows

def centered_crop_extent(valid_union: Bounds, raster_union: Bounds, target_aspect: float, padding_ratio: float=0.03, min_padding_m: float=2.0, clip_to_bounds: bool=True) -> tuple[Bounds, list[str]]:
    warnings: list[str] = []
    pad = max(max(valid_union.width, valid_union.height) * padding_ratio, min_padding_m)
    left = valid_union.left - pad
    right = valid_union.right + pad
    bottom = valid_union.bottom - pad
    top = valid_union.top + pad
    width = right - left
    height = top - bottom
    current = width / height
    cx = (left + right) / 2.0
    cy = (bottom + top) / 2.0
    if current < target_aspect:
        width = height * target_aspect
    else:
        height = width / target_aspect
    crop = Bounds(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)
    if not clip_to_bounds:
        return (crop, warnings)
    (crop, clip_warnings) = clip_crop_to_bounds(crop, raster_union)
    warnings.extend(clip_warnings)
    return (crop, warnings)

def clip_crop_to_bounds(crop: Bounds, limit: Bounds) -> tuple[Bounds, list[str]]:
    warnings: list[str] = []
    (left, right) = (crop.left, crop.right)
    (bottom, top) = (crop.bottom, crop.top)
    (width, height) = (crop.width, crop.height)
    if left < limit.left:
        right += limit.left - left
        left = limit.left
        warnings.append('crop shifted right at raster left bound')
    if right > limit.right:
        left -= right - limit.right
        right = limit.right
        warnings.append('crop shifted left at raster right bound')
    if bottom < limit.bottom:
        top += limit.bottom - bottom
        bottom = limit.bottom
        warnings.append('crop shifted up at raster bottom bound')
    if top > limit.top:
        bottom -= top - limit.top
        top = limit.top
        warnings.append('crop shifted down at raster top bound')
    if left < limit.left:
        left = limit.left
        warnings.append('crop width clipped at raster bounds')
    if right > limit.right:
        right = limit.right
        warnings.append('crop width clipped at raster bounds')
    if bottom < limit.bottom:
        bottom = limit.bottom
        warnings.append('crop height clipped at raster bounds')
    if top > limit.top:
        top = limit.top
        warnings.append('crop height clipped at raster bounds')
    if right - left < width * 0.99 or top - bottom < height * 0.99:
        warnings.append('crop aspect/centering changed by raster-bound clipping')
    return (Bounds(left, bottom, right, top), warnings)

def utm46n_to_lonlat(x: float, y: float) -> tuple[float, float]:
    zone_number = 46
    lon_origin = (zone_number - 1) * 6 - 180 + 3
    k0 = 0.9996
    e1 = (1 - math.sqrt(1 - WGS84_E2)) / (1 + math.sqrt(1 - WGS84_E2))
    x_adj = x - 500000.0
    m = y / k0
    mu = m / (WGS84_A * (1 - WGS84_E2 / 4 - 3 * WGS84_E2 ** 2 / 64 - 5 * WGS84_E2 ** 3 / 256))
    j1 = 3 * e1 / 2 - 27 * e1 ** 3 / 32
    j2 = 21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32
    j3 = 151 * e1 ** 3 / 96
    j4 = 1097 * e1 ** 4 / 512
    fp = mu + j1 * math.sin(2 * mu) + j2 * math.sin(4 * mu) + j3 * math.sin(6 * mu) + j4 * math.sin(8 * mu)
    e2_prime = WGS84_E2 / (1 - WGS84_E2)
    c1 = e2_prime * math.cos(fp) ** 2
    t1 = math.tan(fp) ** 2
    n1 = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(fp) ** 2)
    r1 = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * math.sin(fp) ** 2) ** 1.5
    d = x_adj / (n1 * k0)
    lat = fp - n1 * math.tan(fp) / r1 * (d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * e2_prime) * d ** 4 / 24 + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * e2_prime - 3 * c1 ** 2) * d ** 6 / 720)
    lon = math.radians(lon_origin) + (d - (1 + 2 * t1 + c1) * d ** 3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * e2_prime + 24 * t1 ** 2) * d ** 5 / 120) / math.cos(fp)
    return (math.degrees(lon), math.degrees(lat))

def lonlat_converter(crs: str):
    normalized = (crs or '').upper().replace(' ', '')
    if Transformer is not None and crs:
        try:
            transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
            return (lambda x, y: transformer.transform(x, y), 'pyproj')
        except Exception:
            pass
    if '32646' in normalized:
        return (utm46n_to_lonlat, 'utm46n_formula')
    raise RuntimeError(f'Unsupported CRS for Longitude/Latitude tick labels: {crs}')

def lonlat_decimals(crop: Bounds, converter) -> int:
    (lon0, lat0) = converter(crop.left, crop.bottom)
    (lon1, lat1) = converter(crop.right, crop.top)
    span = max(abs(lon1 - lon0), abs(lat1 - lat0))
    return 5 if span < 0.02 else 4

def format_lonlat_axes(ax, crop: Bounds, crs: str, show_labels: bool=True, nbins: int=4) -> str:
    import matplotlib.ticker as ticker
    (converter, mode) = lonlat_converter(crs)
    ax.set_xlim(crop.left, crop.right)
    ax.set_ylim(crop.bottom, crop.top)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins))
    decimals = lonlat_decimals(crop, converter)
    xticks = [t for t in ax.get_xticks() if crop.left <= t <= crop.right]
    yticks = [t for t in ax.get_yticks() if crop.bottom <= t <= crop.top]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{converter(float(x), crop.center_y)[0]:.{decimals}f} E' for x in xticks])
    ax.set_yticklabels([f'{converter(crop.center_x, float(y))[1]:.{decimals}f} N' for y in yticks])
    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_va('center')
        label.set_ha('center')
        label.set_rotation_mode('anchor')
    ax.tick_params(axis='x', labelsize=6.6, length=2.5, pad=2)
    ax.tick_params(axis='y', labelsize=6.6, length=2.5, pad=7)
    if show_labels:
        ax.set_xlabel('Longitude', fontsize=7.8, labelpad=5)
        ax.set_ylabel('Latitude', fontsize=7.8, labelpad=17)
        ax.xaxis.set_label_coords(0.5, -0.085)
        ax.yaxis.set_label_coords(-0.145, 0.5)
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(MORANDI_INK)
        spine.set_linewidth(0.55)
    return mode

def reserve_bottom_annotation_margin(crop: Bounds, valid_union: Bounds, margin_fraction: float=0.11, min_margin_m: float=18.0) -> tuple[Bounds, list[str]]:
    target_by_fraction = (valid_union.bottom - margin_fraction * crop.top) / max(1.0 - margin_fraction, 1e-09)
    target_by_minimum = valid_union.bottom - min_margin_m
    new_bottom = min(crop.bottom, target_by_fraction, target_by_minimum)
    if new_bottom < crop.bottom:
        return (Bounds(crop.left, new_bottom, crop.right, crop.top), ['crop expanded downward to reserve a lower-right scale-bar margin outside the slump footprint'])
    return (crop, [])

def fixed_size_centered_crop(center_extent: Bounds, width: float, height: float, raster_bounds: Bounds, clip_to_bounds: bool=True) -> tuple[Bounds, list[str]]:
    crop = Bounds(center_extent.center_x - width / 2.0, center_extent.center_y - height / 2.0, center_extent.center_x + width / 2.0, center_extent.center_y + height / 2.0)
    if clip_to_bounds:
        return clip_crop_to_bounds(crop, raster_bounds)
    return (crop, [])

def add_scale_bar(ax, crop: Bounds, color: str=MORANDI_INK, length_m: float | None=None) -> None:
    width = crop.width
    height = crop.height
    length = float(length_m) if length_m is not None else nice_scale_length(width)
    x0 = crop.right - 0.06 * width - length
    y0 = crop.bottom + 0.045 * height
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=2.0, solid_capstyle='butt', zorder=5)
    ax.plot([x0, x0], [y0 - 0.006 * height, y0 + 0.006 * height], color=color, lw=0.8, zorder=5)
    ax.plot([x0 + length, x0 + length], [y0 - 0.006 * height, y0 + 0.006 * height], color=color, lw=0.8, zorder=5)
    ax.text(x0 + length / 2.0, y0 + 0.025 * height, scale_label(length), ha='center', va='bottom', fontsize=7.5, color=color, zorder=6)

def add_north_arrow(ax, crop: Bounds, color: str=MORANDI_INK) -> None:
    width = crop.width
    height = crop.height
    x = crop.right - 0.085 * width
    y0 = crop.bottom + 0.105 * height
    y1 = y0 + 0.105 * height
    ax.plot([x, x], [y0, y1], color=color, lw=1.1, zorder=6)
    ax.plot([x], [y1], marker='^', markersize=5.5, color=color, zorder=6)
    ax.text(x, y1 + 0.018 * height, 'N', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color, zorder=6)

def nice_scale_length(width_m: float) -> float:
    target = max(width_m * 0.18, 1.0)
    exponent = math.floor(math.log10(target))
    base = target / 10 ** exponent
    if base < 1.5:
        nice = 1.0
    elif base < 3.5:
        nice = 2.0
    elif base < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * 10 ** exponent

def scale_label(length_m: float) -> str:
    if length_m >= 1000:
        return f'{length_m / 1000:g} km'
    return f'{int(round(length_m))} m' if abs(length_m - round(length_m)) < 1e-06 else f'{length_m:g} m'

def extent_table_rows(layers: list[RasterLayer], las_files: list[Path], crop: Bounds, tolerance_m: float) -> list[dict[str, Any]]:
    rows = []
    for (layer, las_path) in zip(layers, las_files):
        las_bounds = read_las_header_bounds(las_path)
        deltas = extent_deltas(layer.valid_extent, las_bounds)
        rows.append({'label': layer.label, 'raster_path': str(layer.path), 'las_path': str(las_path.resolve()), 'crs': layer.crs, 'raster_valid_left': layer.valid_extent.left, 'raster_valid_bottom': layer.valid_extent.bottom, 'raster_valid_right': layer.valid_extent.right, 'raster_valid_top': layer.valid_extent.top, 'las_left': las_bounds.left, 'las_bottom': las_bounds.bottom, 'las_right': las_bounds.right, 'las_top': las_bounds.top, 'delta_max_m': deltas['max_m'], 'within_tolerance': deltas['max_m'] <= tolerance_m, 'crop_left': crop.left, 'crop_bottom': crop.bottom, 'crop_right': crop.right, 'crop_top': crop.top})
    return rows

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')
