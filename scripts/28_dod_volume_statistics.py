from __future__ import annotations
import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import laspy
from scipy.stats import gaussian_kde
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from map_render_utils import parse_pair_from_name, pair_label, selected_manifest_files, sort_dod_files, write_json
from utils_project_paths import get_data_source, list_latest_files, load_manifest, write_input_audit
SCRIPT_NAME = Path(__file__).name
OUTPUT_SUBDIR = 'DOD_VOLUME_MULTIANGLE'
PAIR_ORDER = ['240630-230924', '250816-240630', '251017-250816', '250816-230924', '251017-240630', '251017-230924']
MORANDI = {'ink': '#2F3437', 'erosion': '#5B7C99', 'erosion_light': '#8FA5B8', 'deposition': '#B98270', 'deposition_light': '#D3B6A9', 'gross': '#6E9A8D', 'net': '#8E879F', 'stable': '#E7E5E0', 'ochre': '#C2A46D', 'sage': '#9AAA8B', 'gray': '#B8B0A6', 'panel': '#F7F6F2'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8, 'axes.linewidth': 0.55, 'axes.edgecolor': MORANDI['ink'], 'axes.labelcolor': MORANDI['ink'], 'xtick.color': MORANDI['ink'], 'ytick.color': MORANDI['ink'], 'xtick.major.width': 0.45, 'ytick.major.width': 0.45, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})

@dataclass(frozen=True)
class RasterPayload:
    pair: str
    dod_path: Path
    lod_path: Path
    dod: np.ndarray
    lod95: np.ndarray
    valid: np.ndarray
    cell_area_m2: float
    resolution_x_m: float
    resolution_y_m: float
    crs: str
    transform: Any

def pair_sort_key(pair: str) -> tuple[int, str]:
    return (PAIR_ORDER.index(pair) if pair in PAIR_ORDER else 999, pair)

def parse_date_token(token: str) -> pd.Timestamp:
    return pd.Timestamp(year=2000 + int(token[:2]), month=int(token[2:4]), day=int(token[4:6]))

def pair_dates(pair: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    (later, earlier) = pair.split('-')
    return (parse_date_token(later), parse_date_token(earlier))

def discover_inputs(manifest: dict[str, Any]) -> tuple[list[Path], dict[str, Path], list[Path]]:
    dod_tifs = selected_manifest_files('standardized_dod_root', 'standardized_dod_tif', 6, 'dod', manifest=manifest)
    dod_las = selected_manifest_files('standardized_dod_root', 'standardized_dod_las', 6, 'dod', manifest=manifest)
    lod_files = sort_dod_files(list_latest_files('lod95_root', 'lod95', manifest=manifest))
    if len(lod_files) != 6:
        raise RuntimeError(f'Expected 6 LOD95 rasters, found {len(lod_files)}: {[p.name for p in lod_files]}')
    lod_map = {parse_pair_from_name(path): path for path in lod_files}
    tif_pairs = {parse_pair_from_name(path) for path in dod_tifs}
    las_pairs = {parse_pair_from_name(path) for path in dod_las}
    if tif_pairs != las_pairs:
        raise RuntimeError(f'DoD TIF/LAS pair mismatch: tif={sorted(tif_pairs)}, las={sorted(las_pairs)}')
    missing_lod = tif_pairs.difference(lod_map)
    if missing_lod:
        raise RuntimeError(f'Missing LOD95 rasters for pairs: {sorted(missing_lod)}')
    return (dod_tifs, lod_map, dod_las)

def check_same_grid(pair: str, src, lod_src) -> None:
    if src.shape != lod_src.shape:
        raise RuntimeError(f'LOD95 shape mismatch for {pair}: {src.shape} vs {lod_src.shape}')
    if src.crs != lod_src.crs:
        raise RuntimeError(f'LOD95 CRS mismatch for {pair}: {src.crs} vs {lod_src.crs}')
    if not src.transform.almost_equals(lod_src.transform, precision=8):
        raise RuntimeError(f'LOD95 transform mismatch for {pair}')

def read_raster_payload(dod_path: Path, lod_path: Path) -> RasterPayload:
    pair = parse_pair_from_name(dod_path)
    with rasterio.open(dod_path) as src, rasterio.open(lod_path) as lod_src:
        check_same_grid(pair, src, lod_src)
        dod = src.read(1, masked=True).astype('float64').filled(np.nan)
        lod = lod_src.read(1, masked=True).astype('float64').filled(np.nan)
        if src.nodata is not None:
            dod[np.isclose(dod, float(src.nodata), rtol=0.0, atol=1e-09)] = np.nan
        if lod_src.nodata is not None:
            lod[np.isclose(lod, float(lod_src.nodata), rtol=0.0, atol=1e-09)] = np.nan
        valid = np.isfinite(dod) & np.isfinite(lod) & (lod >= 0)
        return RasterPayload(pair=pair, dod_path=dod_path.resolve(), lod_path=lod_path.resolve(), dod=dod, lod95=lod, valid=valid, cell_area_m2=float(abs(src.res[0] * src.res[1])), resolution_x_m=float(abs(src.res[0])), resolution_y_m=float(abs(src.res[1])), crs=str(src.crs) if src.crs is not None else '', transform=src.transform)

def value_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype='float64')
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'sample_count': 0, 'mean_m': math.nan, 'median_m': math.nan, 'std_m': math.nan, 'p05_m': math.nan, 'p25_m': math.nan, 'p75_m': math.nan, 'p95_m': math.nan, 'min_m': math.nan, 'max_m': math.nan}
    return {'sample_count': int(values.size), 'mean_m': float(np.mean(values)), 'median_m': float(np.median(values)), 'std_m': float(np.std(values, ddof=1)) if values.size > 1 else 0.0, 'p05_m': float(np.percentile(values, 5)), 'p25_m': float(np.percentile(values, 25)), 'p75_m': float(np.percentile(values, 75)), 'p95_m': float(np.percentile(values, 95)), 'min_m': float(np.min(values)), 'max_m': float(np.max(values))}

def sampled(values: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype='float64')
    values = values[np.isfinite(values)]
    if values.size <= max_n:
        return values.copy()
    idx = rng.choice(values.size, size=max_n, replace=False)
    return values[idx]

def raster_rows(payload: RasterPayload, zero_tolerance_m: float) -> list[dict[str, Any]]:
    pair = payload.pair
    (later, earlier) = pair_dates(pair)
    valid = payload.valid
    dod = payload.dod
    lod = payload.lod95
    area = payload.cell_area_m2
    raw_erosion = valid & (dod < -zero_tolerance_m)
    raw_deposition = valid & (dod > zero_tolerance_m)
    raw_stable = valid & ~(raw_erosion | raw_deposition)
    sig_erosion = valid & (dod < -lod)
    sig_deposition = valid & (dod > lod)
    sig_any = sig_erosion | sig_deposition
    sig_stable = valid & ~sig_any
    lod_values = lod[valid]
    lod_min = float(np.nanmin(lod_values)) if lod_values.size else math.nan
    lod_max = float(np.nanmax(lod_values)) if lod_values.size else math.nan
    common = {'pair': pair, 'period': pair_label(pair), 'later_date': later.date().isoformat(), 'earlier_date': earlier.date().isoformat(), 'interval_days': int((later - earlier).days), 'sign_convention': 'later - earlier; negative=erosion/lowering; positive=deposition/raising', 'source_type': 'tif_raster', 'dod_tif': str(payload.dod_path), 'lod95_tif': str(payload.lod_path), 'cell_area_m2': payload.cell_area_m2, 'resolution_x_m': payload.resolution_x_m, 'resolution_y_m': payload.resolution_y_m, 'crs': payload.crs, 'valid_cell_count': int(np.count_nonzero(valid)), 'valid_area_m2': float(np.count_nonzero(valid) * area), 'lod95_threshold_m_for_las': float(np.nanmedian(lod_values)) if lod_values.size else math.nan, 'lod95_mean_m': float(np.nanmean(lod_values)) if lod_values.size else math.nan, 'lod95_median_m': float(np.nanmedian(lod_values)) if lod_values.size else math.nan, 'lod95_min_m': lod_min, 'lod95_max_m': lod_max, 'lod95_is_spatially_variable': bool(np.isfinite(lod_min) and np.isfinite(lod_max) and (abs(lod_max - lod_min) > 1e-06))}
    rows = []
    for (scope, erosion_mask, deposition_mask, stable_mask, stats_mask, threshold_note) in [('raw_unfiltered', raw_erosion, raw_deposition, raw_stable, valid, f'DoD < -{zero_tolerance_m:g} / DoD > {zero_tolerance_m:g}'), ('lod95_filtered', sig_erosion, sig_deposition, sig_stable, sig_any, 'DoD < -LOD95 / DoD > LOD95')]:
        erosion_volume = float(np.nansum(dod[erosion_mask]) * area)
        deposition_volume = float(np.nansum(dod[deposition_mask]) * area)
        row = {**common, 'stat_scope': scope, 'classification_rule': threshold_note, 'erosion_volume_m3': erosion_volume, 'deposition_volume_m3': deposition_volume, 'net_volume_m3': erosion_volume + deposition_volume, 'gross_volume_m3': abs(erosion_volume) + deposition_volume, 'erosion_area_m2': float(np.count_nonzero(erosion_mask) * area), 'deposition_area_m2': float(np.count_nonzero(deposition_mask) * area), 'stable_area_m2': float(np.count_nonzero(stable_mask) * area), 'active_area_m2': float((np.count_nonzero(erosion_mask) + np.count_nonzero(deposition_mask)) * area), 'erosion_cell_count': int(np.count_nonzero(erosion_mask)), 'deposition_cell_count': int(np.count_nonzero(deposition_mask)), 'stable_cell_count': int(np.count_nonzero(stable_mask))}
        row.update(value_stats(dod[stats_mask]))
        rows.append(row)
    return rows

def lod_threshold_from_payload(payload: RasterPayload) -> float:
    vals = payload.lod95[payload.valid]
    return float(np.nanmedian(vals)) if vals.size else math.nan

def las_rows_and_sample(las_path: Path, payload: RasterPayload, zero_tolerance_m: float, sample_n: int, rng: np.random.Generator) -> tuple[list[dict[str, Any]], np.ndarray]:
    pair = parse_pair_from_name(las_path)
    (later, earlier) = pair_dates(pair)
    threshold = lod_threshold_from_payload(payload)
    las = laspy.read(las_path)
    z = np.asarray(las.z, dtype='float64')
    z = z[np.isfinite(z)]
    raw_erosion = z < -zero_tolerance_m
    raw_deposition = z > zero_tolerance_m
    raw_stable = ~(raw_erosion | raw_deposition)
    sig_erosion = z < -threshold
    sig_deposition = z > threshold
    sig_any = sig_erosion | sig_deposition
    sig_stable = ~sig_any
    bbox_area = float(max((las.header.maxs[0] - las.header.mins[0]) * (las.header.maxs[1] - las.header.mins[1]), 0.0))
    common = {'pair': pair, 'period': pair_label(pair), 'later_date': later.date().isoformat(), 'earlier_date': earlier.date().isoformat(), 'interval_days': int((later - earlier).days), 'sign_convention': 'later - earlier; negative=erosion/lowering; positive=deposition/raising', 'source_type': 'las_point_cloud', 'dod_las': str(las_path.resolve()), 'lod95_threshold_m': threshold, 'point_count_total': int(z.size), 'las_bbox_area_m2': bbox_area, 'las_bbox_point_density_points_m2': float(z.size / bbox_area) if bbox_area > 0 else math.nan, 'las_min_x': float(las.header.mins[0]), 'las_min_y': float(las.header.mins[1]), 'las_max_x': float(las.header.maxs[0]), 'las_max_y': float(las.header.maxs[1])}
    rows = []
    for (scope, erosion_mask, deposition_mask, stable_mask, stats_mask, threshold_note) in [('raw_unfiltered', raw_erosion, raw_deposition, raw_stable, np.ones_like(z, dtype=bool), f'z < -{zero_tolerance_m:g} / z > {zero_tolerance_m:g}'), ('lod95_filtered', sig_erosion, sig_deposition, sig_stable, sig_any, 'z < -median(LOD95) / z > median(LOD95)')]:
        active = erosion_mask | deposition_mask
        row = {**common, 'stat_scope': scope, 'classification_rule': threshold_note, 'active_point_count': int(np.count_nonzero(active)), 'erosion_point_count': int(np.count_nonzero(erosion_mask)), 'deposition_point_count': int(np.count_nonzero(deposition_mask)), 'stable_point_count': int(np.count_nonzero(stable_mask)), 'active_point_fraction_percent': float(100.0 * np.count_nonzero(active) / max(z.size, 1)), 'erosion_point_fraction_percent': float(100.0 * np.count_nonzero(erosion_mask) / max(z.size, 1)), 'deposition_point_fraction_percent': float(100.0 * np.count_nonzero(deposition_mask) / max(z.size, 1)), 'stable_point_fraction_percent': float(100.0 * np.count_nonzero(stable_mask) / max(z.size, 1))}
        row.update(value_stats(z[stats_mask]))
        rows.append(row)
    return (rows, sampled(z, sample_n, rng))

def ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and (not overwrite):
        raise FileExistsError(f'Output exists; use --overwrite to replace it: {path}')

def save_all_formats(fig: plt.Figure, stem: Path, overwrite: bool, dpi: int) -> list[str]:
    outputs = []
    for suffix in ('png', 'svg', 'pdf'):
        path = stem.with_suffix(f'.{suffix}')
        ensure_writable(path, overwrite)
        fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
        outputs.append(str(path))
    plt.close(fig)
    return outputs

def style_axes(ax: plt.Axes, grid: bool=True) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid:
        ax.grid(axis='y', color='#D8D5CD', lw=0.35, alpha=0.75)
    ax.set_axisbelow(True)

def short_pair_labels(pairs: list[str]) -> list[str]:
    return [p.replace('-', '\n') for p in pairs]

def plot_volume_change_bars(df: pd.DataFrame, figure_dir: Path, overwrite: bool, dpi: int) -> list[str]:
    x = np.arange(len(PAIR_ORDER))
    (fig, axes) = plt.subplots(1, 2, figsize=(7.3, 2.8))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.88, wspace=0.28)
    for (ax, metric, ylabel, colors) in [(axes[0], 'net_volume_m3', 'Net volume (m3)', [MORANDI['gray'], MORANDI['net']]), (axes[1], 'gross_volume_m3', 'Gross volume (m3)', [MORANDI['sage'], MORANDI['gross']])]:
        raw_values = df[df['stat_scope'] == 'raw_unfiltered'].set_index('pair').loc[PAIR_ORDER, metric].to_numpy(dtype='float64')
        lod_values = df[df['stat_scope'] == 'lod95_filtered'].set_index('pair').loc[PAIR_ORDER, metric].to_numpy(dtype='float64')
        ax.bar(x - 0.18, raw_values, width=0.34, color=colors[0], label='Raw')
        ax.bar(x + 0.18, lod_values, width=0.34, color=colors[1], label='LOD95')
        ax.axhline(0, color=MORANDI['ink'], lw=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(short_pair_labels(PAIR_ORDER), fontsize=6.2)
        ax.set_ylabel(ylabel)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc='upper right')
    axes[0].text(0.01, 0.98, 'a', transform=axes[0].transAxes, va='top', fontweight='bold')
    axes[1].text(0.01, 0.98, 'b', transform=axes[1].transAxes, va='top', fontweight='bold')
    return save_all_formats(fig, figure_dir / 'FIG05_dod_volume_change_bars', overwrite, dpi)

def plot_erosion_deposition(df: pd.DataFrame, figure_dir: Path, overwrite: bool, dpi: int) -> list[str]:
    (fig, axes) = plt.subplots(1, 2, figsize=(7.3, 2.9), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.86, wspace=0.12)
    x = np.arange(len(PAIR_ORDER))
    for (ax, scope, title) in zip(axes, ['raw_unfiltered', 'lod95_filtered'], ['Raw sign classes', 'LOD95 significant classes']):
        sub = df[df['stat_scope'] == scope].set_index('pair').loc[PAIR_ORDER]
        deposition = sub['deposition_volume_m3'].to_numpy(dtype='float64')
        erosion = sub['erosion_volume_m3'].to_numpy(dtype='float64')
        net = sub['net_volume_m3'].to_numpy(dtype='float64')
        ax.bar(x, deposition, width=0.6, color=MORANDI['deposition'], label='Deposition')
        ax.bar(x, erosion, width=0.6, color=MORANDI['erosion'], label='Erosion')
        ax.plot(x, net, color=MORANDI['ink'], marker='o', ms=3, lw=0.8, label='Net')
        ax.axhline(0, color=MORANDI['ink'], lw=0.55)
        ax.set_title(title, fontsize=8.3)
        ax.set_xticks(x)
        ax.set_xticklabels(short_pair_labels(PAIR_ORDER), fontsize=6.2)
        style_axes(ax)
    axes[0].set_ylabel('Volume (m3)')
    axes[1].legend(frameon=False, fontsize=7, loc='upper right')
    axes[0].text(0.01, 0.98, 'a', transform=axes[0].transAxes, va='top', fontweight='bold')
    axes[1].text(0.01, 0.98, 'b', transform=axes[1].transAxes, va='top', fontweight='bold')
    return save_all_formats(fig, figure_dir / 'FIG05_dod_erosion_deposition_decomposition', overwrite, dpi)

def plot_metric_timeseries(df: pd.DataFrame, metric: str, ylabel: str, stem: str, figure_dir: Path, overwrite: bool, dpi: int) -> list[str]:
    (fig, ax) = plt.subplots(figsize=(4.9, 2.8))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.27, top=0.9)
    x = np.arange(len(PAIR_ORDER))
    for (scope, color, marker, label) in [('raw_unfiltered', MORANDI['gray'], 'o', 'Raw'), ('lod95_filtered', MORANDI['net'] if metric == 'net_volume_m3' else MORANDI['gross'], 's', 'LOD95')]:
        sub = df[df['stat_scope'] == scope].set_index('pair').loc[PAIR_ORDER]
        ax.plot(x, sub[metric].to_numpy(dtype='float64'), marker=marker, ms=3.3, lw=1.1, color=color, label=label)
    ax.axhline(0, color=MORANDI['ink'], lw=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(short_pair_labels(PAIR_ORDER), fontsize=6.3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('DoD pair')
    ax.legend(frameon=False, fontsize=7, ncol=2)
    style_axes(ax)
    return save_all_formats(fig, figure_dir / stem, overwrite, dpi)

def plot_area_volume_scatter(df: pd.DataFrame, figure_dir: Path, overwrite: bool, dpi: int) -> list[str]:
    (fig, ax) = plt.subplots(figsize=(4.2, 3.2))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.92)
    for (scope, color, marker, label) in [('raw_unfiltered', MORANDI['gray'], 'o', 'Raw'), ('lod95_filtered', MORANDI['gross'], 's', 'LOD95')]:
        sub = df[df['stat_scope'] == scope].set_index('pair').loc[PAIR_ORDER]
        active_area = sub['active_area_m2'].to_numpy(dtype='float64')
        gross_volume = sub['gross_volume_m3'].to_numpy(dtype='float64')
        ax.scatter(active_area, gross_volume, s=28, color=color, marker=marker, edgecolor=MORANDI['ink'], lw=0.35, label=label)
        for (pair, row) in sub.iterrows():
            ax.annotate(pair[:6], (row['active_area_m2'], row['gross_volume_m3']), xytext=(3, 3), textcoords='offset points', fontsize=5.8, color=MORANDI['ink'])
    ax.set_xlabel('Active area (m2)')
    ax.set_ylabel('Gross volume (m3)')
    ax.legend(frameon=False, fontsize=7)
    style_axes(ax)
    return save_all_formats(fig, figure_dir / 'FIG05_dod_area_volume_relation', overwrite, dpi)

def kde_line(ax: plt.Axes, values: np.ndarray, color: str, label: str | None=None, lw: float=1.0) -> None:
    values = np.asarray(values, dtype='float64')
    values = values[np.isfinite(values)]
    if values.size < 20 or float(np.nanstd(values)) <= 1e-12:
        return
    (lo, hi) = np.percentile(values, [1, 99])
    xs = np.linspace(lo, hi, 220)
    kde = gaussian_kde(values)
    ax.plot(xs, kde(xs), color=color, lw=lw, label=label)

def draw_violin(ax: plt.Axes, samples: dict[str, np.ndarray]) -> None:
    data = [samples[p][np.isfinite(samples[p])] for p in PAIR_ORDER]
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False, widths=0.75)
    for body in parts['bodies']:
        body.set_facecolor(MORANDI['sage'])
        body.set_edgecolor(MORANDI['ink'])
        body.set_linewidth(0.35)
        body.set_alpha(0.75)
    parts['cmedians'].set_color(MORANDI['ink'])
    parts['cmedians'].set_linewidth(0.75)
    ax.axhline(0, color=MORANDI['ink'], lw=0.55)
    ax.set_xticks(np.arange(1, len(PAIR_ORDER) + 1))
    ax.set_xticklabels(short_pair_labels(PAIR_ORDER), fontsize=6)

def draw_box(ax: plt.Axes, samples: dict[str, np.ndarray]) -> None:
    data = [samples[p][np.isfinite(samples[p])] for p in PAIR_ORDER]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.62)
    for patch in bp['boxes']:
        patch.set(facecolor=MORANDI['deposition_light'], edgecolor=MORANDI['ink'], linewidth=0.45)
    for key in ['whiskers', 'caps', 'medians']:
        for item in bp[key]:
            item.set(color=MORANDI['ink'], linewidth=0.55)
    ax.axhline(0, color=MORANDI['ink'], lw=0.55)
    ax.set_xticklabels(short_pair_labels(PAIR_ORDER), fontsize=6)

def plot_distribution(samples: dict[str, np.ndarray], source_label: str, stem: str, figure_dir: Path, overwrite: bool, dpi: int) -> list[str]:
    colors = [MORANDI['erosion'], MORANDI['deposition'], MORANDI['gross'], MORANDI['ochre'], MORANDI['net'], MORANDI['sage']]
    all_values = np.concatenate([samples[p] for p in PAIR_ORDER if samples[p].size])
    clip = float(max(np.nanpercentile(np.abs(all_values), 99), 0.1)) if all_values.size else 1.0
    (fig, axes) = plt.subplots(2, 2, figsize=(7.2, 5.4))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.92, hspace=0.34, wspace=0.28)
    bins = np.linspace(-clip, clip, 80)
    axes[0, 0].hist(np.clip(all_values, -clip, clip), bins=bins, color=MORANDI['gray'], alpha=0.85, density=True)
    kde_line(axes[0, 0], np.clip(all_values, -clip, clip), MORANDI['ink'], lw=1.1)
    axes[0, 0].axvline(0, color=MORANDI['ink'], lw=0.55)
    axes[0, 0].set_title('Histogram and KDE', fontsize=8.3)
    axes[0, 0].set_xlabel('DoD dz (m)')
    axes[0, 0].set_ylabel('Density')
    for (pair, color) in zip(PAIR_ORDER, colors):
        vals = np.clip(samples[pair], -clip, clip)
        kde_line(axes[0, 1], vals, color, pair, lw=0.9)
    axes[0, 1].axvline(0, color=MORANDI['ink'], lw=0.55)
    axes[0, 1].set_title('Pairwise KDE', fontsize=8.3)
    axes[0, 1].set_xlabel('DoD dz (m)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend(frameon=False, fontsize=5.7, ncol=1)
    draw_violin(axes[1, 0], {p: np.clip(samples[p], -clip, clip) for p in PAIR_ORDER})
    axes[1, 0].set_title('Violin plot', fontsize=8.3)
    axes[1, 0].set_ylabel('DoD dz (m)')
    draw_box(axes[1, 1], {p: np.clip(samples[p], -clip, clip) for p in PAIR_ORDER})
    axes[1, 1].set_title('Box plot', fontsize=8.3)
    axes[1, 1].set_ylabel('DoD dz (m)')
    for (label, ax) in zip(['a', 'b', 'c', 'd'], axes.flat):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, va='top', fontweight='bold')
        style_axes(ax)
    fig.suptitle(f'{source_label} DoD distribution diagnostics', fontsize=9.5, color=MORANDI['ink'])
    return save_all_formats(fig, figure_dir / stem, overwrite, dpi)

def write_sample_table(samples: dict[str, np.ndarray], path: Path, overwrite: bool) -> str:
    ensure_writable(path, overwrite)
    rows = []
    for pair in PAIR_ORDER:
        values = samples.get(pair, np.array([], dtype='float64'))
        rows.extend(({'pair': pair, 'period': pair_label(pair), 'dz_m': float(value)} for value in values[np.isfinite(values)]))
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    return str(path)

def write_tables(raster_df: pd.DataFrame, las_df: pd.DataFrame, figure_manifest: dict[str, Any], table_dir: Path, overwrite: bool) -> dict[str, str]:
    paths = {'raster_csv': table_dir / 'TABLE03_dod_raster_volume_multiangle.csv', 'las_csv': table_dir / 'TABLE04_dod_las_point_stats.csv', 'excel': table_dir / 'TABLE03_TABLE04_dod_tif_las_multiangle_stats.xlsx'}
    for path in paths.values():
        ensure_writable(path, overwrite)
    raster_df.to_csv(paths['raster_csv'], index=False, encoding='utf-8-sig')
    las_df.to_csv(paths['las_csv'], index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(paths['excel'], engine='openpyxl') as writer:
        raster_df.to_excel(writer, sheet_name='TIF_raster_budget', index=False)
        las_df.to_excel(writer, sheet_name='LAS_point_stats', index=False)
        pd.DataFrame(figure_manifest['figures']).to_excel(writer, sheet_name='Figure_outputs', index=False)
    return {key: str(path) for (key, path) in paths.items()}

def fmt_num(value: float, unit: str='') -> str:
    if not np.isfinite(value):
        return 'NA'
    return f'{value:,.1f}{unit}'

def write_markdown(raster_df: pd.DataFrame, las_df: pd.DataFrame, outputs: dict[str, Any], text_dir: Path, overwrite: bool) -> str:
    path = text_dir / 'RESULTS_DOD_VOLUME_MULTIANGLE.md'
    ensure_writable(path, overwrite)
    lod = raster_df[raster_df['stat_scope'] == 'lod95_filtered'].copy()
    raw = raster_df[raster_df['stat_scope'] == 'raw_unfiltered'].copy()
    max_gross = lod.loc[lod['gross_volume_m3'].idxmax()]
    max_active = lod.loc[lod['active_area_m2'].idxmax()]
    latest = lod[lod['pair'] == '251017-250816'].iloc[0]
    net_total_lod = float(lod['net_volume_m3'].sum())
    gross_total_lod = float(lod['gross_volume_m3'].sum())
    erosion_total = float(lod['erosion_volume_m3'].sum())
    deposition_total = float(lod['deposition_volume_m3'].sum())
    raw_net_total = float(raw['net_volume_m3'].sum())
    las_lod = las_df[las_df['stat_scope'] == 'lod95_filtered']
    max_las = las_lod.loc[las_lod['active_point_count'].idxmax()]
    figure_lines = '\n'.join((f"- `{item['name']}`: `{item['png']}`, `{item['svg']}`, `{item['pdf']}`" for item in outputs['figures']))
    table_lines = '\n'.join((f'- {key}: `{value}`' for (key, value) in outputs['tables'].items()))
    text = f"Public-release status message.{datetime.now().isoformat(timespec='seconds')}`\n\nScript: `{SCRIPT_NAME}Public-release status message.{fmt_num(erosion_total, ' m3')}Public-release status message.{fmt_num(deposition_total, ' m3')}Public-release status message.{fmt_num(net_total_lod, ' m3')}Public-release status message.{fmt_num(gross_total_lod, ' m3')}Public-release status message.{fmt_num(raw_net_total, ' m3')}Public-release status message.{max_gross['period']}Public-release status message.{fmt_num(float(max_gross['gross_volume_m3']), ' m3')}Public-release status message.{max_active['period']}Public-release status message.{fmt_num(float(max_active['active_area_m2']), ' m2')}Public-release status message.{latest['period']}Public-release status message.{fmt_num(float(latest['net_volume_m3']), ' m3')}Public-release status message.{fmt_num(float(latest['gross_volume_m3']), ' m3')}Public-release status message.{max_las['period']}Public-release status message.{int(max_las['active_point_count']):,}Public-release status message.{int(max_las['erosion_point_count']):,}Public-release status message.{int(max_las['deposition_point_count']):,}Public-release status message.{table_lines}Public-release status message.{figure_lines}\n"
    path.write_text(text, encoding='utf-8')
    return str(path)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compute DoD TIF/LAS volume, area, and distribution statistics with Nature-style figures.')
    parser.add_argument('--manifest', type=Path, default=PROJECT_ROOT / 'DATA_MANIFEST.yaml')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--zero-tolerance', type=float, default=0.0)
    parser.add_argument('--sample-n', type=int, default=80000, help='Maximum distribution-plot sample per pair and source.')
    parser.add_argument('--seed', type=int, default=20260508)
    parser.add_argument('--stats-only', action='store_true', help='Compute tables and distribution samples only; skip Matplotlib figures.')
    return parser

def main() -> None:
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest)
    paper_dir = get_data_source('paper_results_root', manifest=manifest, must_exist=True)
    figure_dir = paper_dir / 'FIGURES' / OUTPUT_SUBDIR
    table_dir = paper_dir / 'TABLES'
    text_dir = paper_dir / 'TEXT'
    for directory in (figure_dir, table_dir, text_dir):
        directory.mkdir(parents=True, exist_ok=True)
    print('Environment: geo')
    print(f"Input root: {get_data_source('standardized_dod_root', manifest=manifest, must_exist=True)}")
    print(f'Output root: {figure_dir}')
    print('Completion standard: CSV/Excel tables, PNG 600 dpi, SVG/PDF figures, Markdown Results/Discussion text, JSON manifest.')
    audit_path = write_input_audit('dod_volume_multiangle_stats', {'standardized_dod_root': ['standardized_dod_tif', 'standardized_dod_las'], 'lod95_root': ['lod95', 'lod95_significant']}, figure_dir, manifest=manifest)
    rng = np.random.default_rng(args.seed)
    (dod_tifs, lod_map, dod_las) = discover_inputs(manifest)
    raster_rows_out: list[dict[str, Any]] = []
    las_rows_out: list[dict[str, Any]] = []
    raster_samples: dict[str, np.ndarray] = {}
    las_samples: dict[str, np.ndarray] = {}
    payloads: dict[str, RasterPayload] = {}
    for dod_path in dod_tifs:
        pair = parse_pair_from_name(dod_path)
        payload = read_raster_payload(dod_path, lod_map[pair])
        payloads[pair] = payload
        raster_rows_out.extend(raster_rows(payload, args.zero_tolerance))
        raster_samples[pair] = sampled(payload.dod[payload.valid], args.sample_n, rng)
    for las_path in dod_las:
        pair = parse_pair_from_name(las_path)
        (rows, sample) = las_rows_and_sample(las_path, payloads[pair], args.zero_tolerance, args.sample_n, rng)
        las_rows_out.extend(rows)
        las_samples[pair] = sample
    raster_df = pd.DataFrame(raster_rows_out).sort_values(['pair', 'stat_scope'], key=lambda s: s.map(lambda v: pair_sort_key(v)[0] if v in PAIR_ORDER else v) if s.name == 'pair' else s)
    raster_df['pair_order'] = raster_df['pair'].map({pair: i for (i, pair) in enumerate(PAIR_ORDER)})
    raster_df = raster_df.sort_values(['pair_order', 'stat_scope']).drop(columns=['pair_order'])
    las_df = pd.DataFrame(las_rows_out)
    las_df['pair_order'] = las_df['pair'].map({pair: i for (i, pair) in enumerate(PAIR_ORDER)})
    las_df = las_df.sort_values(['pair_order', 'stat_scope']).drop(columns=['pair_order'])
    figure_manifest = {'figures': []}
    if not args.stats_only:
        for (name, outputs) in [('FIG05_dod_volume_change_bars', plot_volume_change_bars(raster_df, figure_dir, args.overwrite, args.dpi)), ('FIG05_dod_erosion_deposition_decomposition', plot_erosion_deposition(raster_df, figure_dir, args.overwrite, args.dpi)), ('FIG05_dod_net_volume_timeseries', plot_metric_timeseries(raster_df, 'net_volume_m3', 'Net volume (m3)', 'FIG05_dod_net_volume_timeseries', figure_dir, args.overwrite, args.dpi)), ('FIG05_dod_gross_volume_timeseries', plot_metric_timeseries(raster_df, 'gross_volume_m3', 'Gross volume (m3)', 'FIG05_dod_gross_volume_timeseries', figure_dir, args.overwrite, args.dpi)), ('FIG05_dod_area_volume_relation', plot_area_volume_scatter(raster_df, figure_dir, args.overwrite, args.dpi)), ('FIGS06_dod_raster_distribution_diagnostics', plot_distribution(raster_samples, 'Raster TIF', 'FIGS06_dod_raster_distribution_diagnostics', figure_dir, args.overwrite, args.dpi)), ('FIGS06_dod_las_distribution_diagnostics', plot_distribution(las_samples, 'LAS point-cloud', 'FIGS06_dod_las_distribution_diagnostics', figure_dir, args.overwrite, args.dpi))]:
            figure_manifest['figures'].append({'name': name, 'png': outputs[0], 'svg': outputs[1], 'pdf': outputs[2]})
    table_outputs = write_tables(raster_df, las_df, figure_manifest, table_dir, args.overwrite)
    table_outputs['raster_distribution_sample_csv'] = write_sample_table(raster_samples, table_dir / 'TABLE03_dod_raster_distribution_samples.csv', args.overwrite)
    table_outputs['las_distribution_sample_csv'] = write_sample_table(las_samples, table_dir / 'TABLE04_dod_las_distribution_samples.csv', args.overwrite)
    markdown_path = ''
    if not args.stats_only:
        markdown_path = write_markdown(raster_df, las_df, {'tables': table_outputs, 'figures': figure_manifest['figures']}, text_dir, args.overwrite)
    manifest_path = figure_dir / 'DOD_VOLUME_MULTIANGLE_manifest.json'
    ensure_writable(manifest_path, args.overwrite)
    write_json(manifest_path, {'generated_at': datetime.now().isoformat(timespec='seconds'), 'script': SCRIPT_NAME, 'manifest': str(Path(args.manifest).resolve()), 'input_audit': str(audit_path), 'inputs': {'dod_tif': [str(path.resolve()) for path in dod_tifs], 'dod_las': [str(path.resolve()) for path in dod_las], 'lod95_tif': {pair: str(path.resolve()) for (pair, path) in lod_map.items()}}, 'formulae': {'volume': 'V = sum(DoD_i * pixel_area)', 'erosion_lod95': 'sum(DoD_i < -LOD95 of DoD_i * pixel_area)', 'deposition_lod95': 'sum(DoD_i > LOD95 of DoD_i * pixel_area)', 'net': 'V_deposition + V_erosion', 'gross': 'abs(V_erosion) + V_deposition'}, 'outputs': {'tables': table_outputs, 'markdown': markdown_path, 'figures': figure_manifest['figures']}, 'style': {'target': 'Nature / Nature Communications / Nature Geoscience', 'palette': 'Morandi muted colors; erosion blue, deposition terracotta, stable light gray', 'png_dpi': args.dpi}})
    print('DoD TIF/LAS multi-angle statistics complete.')
    print(f'- Tables: {table_outputs}')
    print(f'- Figures: {figure_dir}')
    if markdown_path:
        print(f'- Markdown: {markdown_path}')
    else:
        print('- Markdown: skipped by --stats-only')
    print(f'- Manifest: {manifest_path}')
if __name__ == '__main__':
    main()
