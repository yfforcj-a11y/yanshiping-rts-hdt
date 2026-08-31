from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from rasterio.plot import plotting_extent
BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / 'outputs'
PAPER_DIR = OUT_DIR / 'paper_results'
FIG_DIR = PAPER_DIR / 'FIGURES'
TABLE_DIR = PAPER_DIR / 'TABLES'
TEXT_DIR = PAPER_DIR / 'TEXT'
for d in (PAPER_DIR, FIG_DIR, TABLE_DIR, TEXT_DIR):
    d.mkdir(parents=True, exist_ok=True)
PAIR_ORDER = ['240630-230924', '250816-230924', '250816-240630', '251017-230924', '251017-240630', '251017-250816']
GNSS_SITES = ['7704', '7627', '9286']
LFJ_SITES = ['3A9', '3D3']
BH_SITES = ['1C', '2C', '3C', '4C', '5C']
CORE_SITES = GNSS_SITES + LFJ_SITES
ROLE_COLORS = {'active': '#b22222', 'stable': '#2b6cb0', 'borehole': '#2f855a', 'other': '#6b7280'}
SITE_MARKERS = {'GNSS': 'o', 'LFJ': 's', 'BH': '^'}

def safe_read_csv(path: Path) -> pd.DataFrame:
    for enc in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def parse_pair_dates(pair: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    (later, earlier) = pair.split('-')
    t0 = pd.Timestamp(year=2000 + int(earlier[:2]), month=int(earlier[2:4]), day=int(earlier[4:6]))
    t1 = pd.Timestamp(year=2000 + int(later[:2]), month=int(later[2:4]), day=int(later[4:6]))
    return (t0, t1)

def pair_label(pair: str) -> str:
    (t0, t1) = parse_pair_dates(pair)
    return f'{t1:%Y-%m-%d}\nvs\n{t0:%Y-%m-%d}'

def figsave(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def get_site_type(site_name: str) -> str:
    if str(site_name).startswith('GNSS'):
        return 'GNSS'
    if str(site_name).startswith('LFJ'):
        return 'LFJ'
    if str(site_name).startswith('BH'):
        return 'BH'
    return 'OTHER'

def compute_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.array([v for v in values if pd.notna(v) and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {'n': 0, 'min': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan, 'range': np.nan}
    return {'n': int(arr.size), 'min': float(np.min(arr)), 'max': float(np.max(arr)), 'mean': float(np.mean(arr)), 'std': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0, 'range': float(np.max(arr) - np.min(arr))}

def build_site_catalog(features: pd.DataFrame) -> pd.DataFrame:
    cat = features.sort_values(['site_id', 'pair']).groupby('site_id', as_index=False).first()[['site_id', 'site_name', 'lon', 'lat', 'x', 'y']].copy()
    cat['site_type'] = cat['site_name'].map(get_site_type)
    cat['field_role'] = np.where(cat['site_type'] == 'BH', 'borehole', 'other')
    return cat

def attach_roles(site_catalog: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    role_map = stability.set_index('sensor')['role'].to_dict() if 'sensor' in stability.columns else {}
    site_catalog['field_role'] = site_catalog.apply(lambda r: role_map.get(r['site_id'], 'borehole' if r['site_type'] == 'BH' else 'other'), axis=1)
    return site_catalog

def compute_nearest_bh(site_catalog: pd.DataFrame) -> pd.DataFrame:
    bh = site_catalog[site_catalog['site_type'] == 'BH'][['site_id', 'x', 'y']].copy()
    out = []
    for (_, row) in site_catalog.iterrows():
        if row['site_type'] == 'BH':
            out.append((row['site_id'], row['site_id'], 0.0))
            continue
        d = np.sqrt((bh['x'] - row['x']) ** 2 + (bh['y'] - row['y']) ** 2)
        idx = int(d.idxmin())
        out.append((row['site_id'], bh.loc[idx, 'site_id'], float(d.loc[idx])))
    nearest = pd.DataFrame(out, columns=['site_id', 'nearest_borehole', 'nearest_bh_distance_m'])
    return site_catalog.merge(nearest, on='site_id', how='left')

def override_gnss_mapping(site_catalog: pd.DataFrame, mapping: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    map_bh = mapping.get('mapping', {})
    map_dist = mapping.get('distance_m', {})
    for (site_id, bh) in map_bh.items():
        site_catalog.loc[site_catalog['site_id'] == site_id, 'nearest_borehole'] = bh
        site_catalog.loc[site_catalog['site_id'] == site_id, 'nearest_bh_distance_m'] = float(map_dist.get(site_id, np.nan))
    return site_catalog

def build_site_pair_fusion(site_catalog: pd.DataFrame, features: pd.DataFrame, ice_pair: pd.DataFrame, pair_rates: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    use_cols = ['site_id', 'site_name', 'pair', 'dz_point', 'abs_dz_point', 'dz_mean_local', 'LOD_sig', 'SCM_prob', 'SCM_bin', 'dist_to_patch_m']
    base = features[use_cols].copy()
    base = base.merge(site_catalog[['site_id', 'site_type', 'field_role', 'nearest_borehole', 'nearest_bh_distance_m']], on='site_id', how='left')
    pair_thermal = ice_pair.rename(columns={'site_id': 'nearest_borehole'}).copy()
    base = base.merge(pair_thermal[['nearest_borehole', 'pair', 'posT_mean', 'ALT_mean', 'meltcol_mean', 'enthalpy_mean', 'posT_max', 'ALT_max', 'meltcol_max', 'enthalpy_max']], on=['nearest_borehole', 'pair'], how='left')
    for col in ('E_retro_rate', 'L_debris_rate', 'S_shear_rate'):
        if col not in base.columns:
            base[col] = np.nan
    for (site_id, rate_df) in pair_rates.items():
        for col in ('E_retro_rate', 'L_debris_rate', 'S_shear_rate'):
            if col not in rate_df.columns:
                rate_df[col] = np.nan
        merged = rate_df[['pair', 'E_retro_rate', 'L_debris_rate', 'S_shear_rate']].copy()
        mask = base['site_id'] == site_id
        if mask.any():
            sub = base.loc[mask, ['pair']].merge(merged, on='pair', how='left')
            base.loc[mask, 'E_retro_rate'] = sub['E_retro_rate'].values
            base.loc[mask, 'L_debris_rate'] = sub['L_debris_rate'].values
            base.loc[mask, 'S_shear_rate'] = sub['S_shear_rate'].values
    base['pair_order'] = base['pair'].map({p: i for (i, p) in enumerate(PAIR_ORDER)})
    return base.sort_values(['pair_order', 'site_type', 'site_id']).drop(columns=['pair_order'])

def compute_gnss_model_metrics(ts: pd.DataFrame) -> Tuple[float, float]:
    obs = pd.to_numeric(ts['w_obs_m'], errors='coerce')
    pred = pd.to_numeric(ts['w_pred_m'], errors='coerce')
    mask = obs.notna() & pred.notna()
    if mask.sum() < 2:
        return (np.nan, np.nan)
    obs = obs[mask].to_numpy(float)
    pred = pred[mask].to_numpy(float)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    ss_res = np.sum((obs - pred) ** 2)
    r2 = np.nan if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    return (r2, rmse)

def build_site_summary(site_catalog: pd.DataFrame, features: pd.DataFrame, event_times: pd.DataFrame, stability: pd.DataFrame, thermo_dir: Path, event_dir: Path) -> pd.DataFrame:
    rows = []
    event_map = event_times.set_index('sensor').to_dict('index') if 'sensor' in event_times.columns else {}
    stability_map = stability.set_index('sensor').to_dict('index') if 'sensor' in stability.columns else {}
    feat_sel = features[features['site_id'].isin(CORE_SITES)].copy()
    for (_, site) in site_catalog[site_catalog['site_id'].isin(CORE_SITES)].iterrows():
        site_id = site['site_id']
        site_feat = feat_sel[feat_sel['site_id'] == site_id]
        row = {'site_id': site_id, 'site_name': site['site_name'], 'site_type': site['site_type'], 'field_role': site['field_role'], 'nearest_borehole': site['nearest_borehole'], 'nearest_bh_distance_m': site['nearest_bh_distance_m'], 'event_trigger': event_map.get(site_id, {}).get('t_trigger', np.nan), 'event_failure': event_map.get(site_id, {}).get('t_failure', np.nan), 'stability_valid_ratio': stability_map.get(site_id, {}).get('valid_ratio', np.nan), 'stability_fail_max_run': stability_map.get(site_id, {}).get('fail_max_run', np.nan), 'geomorph_pairs': int(site_feat.shape[0]), 'geomorph_dz_nonempty': int(site_feat['dz_point'].notna().sum()), 'geomorph_dz_min_m': pd.to_numeric(site_feat['dz_point'], errors='coerce').min(), 'geomorph_dz_max_m': pd.to_numeric(site_feat['dz_point'], errors='coerce').max(), 'geomorph_scm_prob_max': pd.to_numeric(site_feat['SCM_prob'], errors='coerce').max(), 'geomorph_lod_sig_count': int((site_feat['LOD_sig'].astype(str) == '1.0').sum()), 'geomorph_scm_bin_count': int((site_feat['SCM_bin'].astype(str) == '1.0').sum()), 'geomorph_patch_dist_min_m': pd.to_numeric(site_feat['dist_to_patch_m'], errors='coerce').min(), 'geomorph_patch_dist_max_m': pd.to_numeric(site_feat['dist_to_patch_m'], errors='coerce').max()}
        if site['site_type'] == 'GNSS':
            ts = safe_read_csv(thermo_dir / f'GNSS_{site_id}_thermo_slump_final_timeseries.csv')
            vals = pd.to_numeric(ts['w_obs_m'], errors='coerce')
            stats = compute_stats(vals)
            (r2, rmse) = compute_gnss_model_metrics(ts)
            row.update({'signal_min_m': stats['min'], 'signal_max_m': stats['max'], 'signal_std_m': stats['std'], 'signal_range_m': stats['range'], 'model_r2': r2, 'model_rmse_m': rmse})
        elif site['site_type'] == 'LFJ':
            ts = safe_read_csv(event_dir / f'LFJ_{site_id}_timeseries_clean_v5.csv')
            open_stats = compute_stats(pd.to_numeric(ts['open_mm'], errors='coerce'))
            rate_stats = compute_stats(pd.to_numeric(ts['open_rate_smooth'], errors='coerce'))
            row.update({'signal_min_open_mm': open_stats['min'], 'signal_max_open_mm': open_stats['max'], 'signal_std_open_mm': open_stats['std'], 'signal_range_open_mm': open_stats['range'], 'signal_min_open_rate_smooth': rate_stats['min'], 'signal_max_open_rate_smooth': rate_stats['max'], 'signal_std_open_rate_smooth': rate_stats['std']})
        rows.append(row)
    return pd.DataFrame(rows)

def build_pair_summary(geomorph_budget: pd.DataFrame, ice_pair: pd.DataFrame, event_times: pd.DataFrame, site_pair_fusion: pd.DataFrame) -> pd.DataFrame:
    g = geomorph_budget.copy()
    g['pair_order'] = g['pair'].map({p: i for (i, p) in enumerate(PAIR_ORDER)})
    g = g.sort_values('pair_order').drop(columns=['pair_order'])
    for bh in ('2C', '5C'):
        sub = ice_pair[ice_pair['site_id'] == bh][['pair', 'ALT_mean', 'meltcol_mean', 'enthalpy_mean']].copy()
        sub = sub.rename(columns={'ALT_mean': f'ALT_mean_{bh}', 'meltcol_mean': f'meltcol_mean_{bh}', 'enthalpy_mean': f'enthalpy_mean_{bh}'})
        g = g.merge(sub, on='pair', how='left')
    g['event_failure_count'] = 0
    g['event_trigger_count'] = 0
    g['active_events'] = ''
    for (idx, row) in g.iterrows():
        (t0, t1) = parse_pair_dates(row['pair'])
        failures = []
        triggers = []
        for (_, ev) in event_times.iterrows():
            sensor = str(ev.get('sensor', ''))
            tf = pd.to_datetime(ev.get('t_failure'), errors='coerce')
            tt = pd.to_datetime(ev.get('t_trigger'), errors='coerce')
            if pd.notna(tf) and t0 <= tf <= t1:
                failures.append(sensor)
            if pd.notna(tt) and t0 <= tt <= t1:
                triggers.append(sensor)
        g.at[idx, 'event_failure_count'] = len(failures)
        g.at[idx, 'event_trigger_count'] = len(triggers)
        g.at[idx, 'active_events'] = ';'.join(failures + triggers)
    for site_id in ('7627', '9286', '3A9', '3D3', '7704'):
        sub = site_pair_fusion[site_pair_fusion['site_id'] == site_id][['pair', 'dz_point', 'SCM_prob', 'dist_to_patch_m', 'LOD_sig']].copy()
        sub = sub.rename(columns={'dz_point': f'{site_id}_dz_point_m', 'SCM_prob': f'{site_id}_SCM_prob', 'dist_to_patch_m': f'{site_id}_dist_to_patch_m', 'LOD_sig': f'{site_id}_LOD_sig'})
        g = g.merge(sub, on='pair', how='left')
    return g

def read_raster(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile
    return (arr, profile)

def discover_raster(root: Path, pattern: str) -> Dict[str, Path]:
    out = {}
    for p in root.rglob(pattern):
        m = re.search('(\\d{6}-\\d{6})', p.name)
        release = re.search('(\\d{8})_minus_(\\d{8})', p.name)
        if m:
            pair = m.group(1)
        elif release:
            pair = f'{release.group(1)[2:]}-{release.group(2)[2:]}'
        else:
            continue
        if pair not in out or len(str(p)) < len(str(out[pair])):
            out[pair] = p
    return out

def export_latest_rasters(site_catalog: pd.DataFrame, scm_path: Path, dod_path: Path) -> None:
    (scm, scm_prof) = read_raster(scm_path)
    (dod, dod_prof) = read_raster(dod_path)
    scm_f = scm.astype(np.float32)
    dod_f = dod.astype(np.float32)
    scm_nodata = scm_prof.get('nodata', None)
    dod_nodata = dod_prof.get('nodata', None)
    if scm_nodata is not None:
        scm_f[scm_f == scm_nodata] = np.nan
    if dod_nodata is not None:
        dod_f[dod_f == dod_nodata] = np.nan
    dod_on_scm = np.full(scm_f.shape, np.nan, dtype=np.float32)
    reproject(source=dod_f, destination=dod_on_scm, src_transform=dod_prof['transform'], src_crs=dod_prof['crs'], src_nodata=np.nan, dst_transform=scm_prof['transform'], dst_crs=scm_prof['crs'], dst_nodata=np.nan, resampling=Resampling.nearest)
    scm_valid = np.isfinite(scm_f)
    dod_on_scm[~scm_valid] = np.nan
    np.savez_compressed(TABLE_DIR / 'latest_pair_rasters.npz', scm=scm_f, dod=dod_on_scm, scm_transform=np.array(tuple(scm_prof['transform'])), dod_transform=np.array(tuple(scm_prof['transform'])))
    site_catalog.to_csv(TABLE_DIR / 'latest_pair_sites_for_map.csv', index=False, encoding='utf-8-sig')

def plot_network_map(site_catalog: pd.DataFrame, scm_path: Path, dod_path: Path) -> None:
    (scm, scm_prof) = read_raster(scm_path)
    (dod, dod_prof) = read_raster(dod_path)
    (fig, axes) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ext1 = plotting_extent(scm, scm_prof['transform'])
    im1 = axes[0].imshow(scm, extent=ext1, cmap='viridis', vmin=0.0, vmax=1.0)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.02, label='SCM probability')
    axes[0].set_title('Latest long-window SCM probability')
    valid_dod = dod[np.isfinite(dod)]
    clip = np.nanquantile(np.abs(valid_dod), 0.98) if valid_dod.size else 1.0
    ext2 = plotting_extent(dod, dod_prof['transform'])
    im2 = axes[1].imshow(dod, extent=ext2, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0.0, vmin=-clip, vmax=clip))
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.02, label='DoD elevation change (m)')
    axes[1].set_title('Latest long-window DoD')
    for ax in axes:
        for (_, row) in site_catalog.iterrows():
            stype = row['site_type']
            role = row['field_role']
            color = ROLE_COLORS.get(role, ROLE_COLORS['other'])
            edge = 'black' if stype != 'BH' else 'white'
            size = 80 if stype != 'BH' else 90
            ax.scatter(row['x'], row['y'], marker=SITE_MARKERS.get(stype, 'o'), s=size, c=color, edgecolors=edge, linewidths=0.8, zorder=3)
            ax.text(row['x'] + 3.0, row['y'] + 3.0, row['site_id'], fontsize=8, color='black', zorder=4)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
    cat_index = site_catalog.set_index('site_id')
    for (left, right) in [('7627', '5C'), ('9286', '2C'), ('7704', '2C')]:
        if left in cat_index.index and right in cat_index.index:
            x = [cat_index.at[left, 'x'], cat_index.at[right, 'x']]
            y = [cat_index.at[left, 'y'], cat_index.at[right, 'y']]
            axes[0].plot(x, y, '--', color='white', linewidth=1.0, alpha=0.8)
            axes[1].plot(x, y, '--', color='black', linewidth=1.0, alpha=0.6)
    legend_items = [Line2D([0], [0], marker='o', color='w', label='GNSS active', markerfacecolor=ROLE_COLORS['active'], markeredgecolor='black', markersize=8), Line2D([0], [0], marker='o', color='w', label='GNSS stable', markerfacecolor=ROLE_COLORS['stable'], markeredgecolor='black', markersize=8), Line2D([0], [0], marker='s', color='w', label='LFJ', markerfacecolor='#8b5cf6', markeredgecolor='black', markersize=8), Line2D([0], [0], marker='^', color='w', label='Borehole', markerfacecolor=ROLE_COLORS['borehole'], markeredgecolor='white', markersize=8)]
    axes[0].legend(handles=legend_items, loc='lower right', framealpha=0.95)
    fig.suptitle('Multisource monitoring network over latest geomorphic-change products', fontsize=14)
    figsave(fig, FIG_DIR / 'FIG01_multisource_network_map.png')

def plot_pair_budget_thermal(pair_summary: pd.DataFrame) -> None:
    df = pair_summary.copy()
    labels = [pair_label(p) for p in df['pair']]
    x = np.arange(len(df))
    (fig, axes) = plt.subplots(3, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    axes[0].bar(x - 0.18, df['vol_erosion_m3'], width=0.35, color='#d62728', label='Erosion volume')
    axes[0].bar(x + 0.18, df['vol_deposition_m3'], width=0.35, color='#1f77b4', label='Deposition volume')
    axes[0].plot(x, df['vol_net_m3'], color='black', marker='o', linewidth=1.8, label='Net volume')
    axes[0].axhline(0.0, color='0.3', linewidth=0.8)
    axes[0].set_ylabel('Volume (m3)')
    axes[0].set_title('Pair-wise geomorphic budget')
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(x, df['mobilized_proxy_m3'], color='#7c3aed', marker='o', linewidth=2.0, label='Mobilized proxy')
    axes[1].plot(x, df['transport_abs_m3'], color='#ff7f0e', marker='s', linewidth=1.7, label='Transport abs volume')
    axes[1].plot(x, df['area_transport_m2'], color='#2ca02c', marker='^', linewidth=1.7, label='Transport area')
    for (xi, txt) in enumerate(df['active_events']):
        if isinstance(txt, str) and txt:
            axes[1].annotate(txt, (xi, df['mobilized_proxy_m3'].iloc[xi]), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
    axes[1].set_ylabel('Geomorphic intensity')
    axes[1].set_title('Mobilization and event overlap')
    axes[1].legend(ncol=3, fontsize=8)
    axes[2].plot(x, df['ALT_mean_2C'], color='#1f77b4', marker='o', linewidth=1.8, label='ALT mean 2C')
    axes[2].plot(x, df['ALT_mean_5C'], color='#d62728', marker='o', linewidth=1.8, label='ALT mean 5C')
    ax2 = axes[2].twinx()
    ax2.plot(x, df['meltcol_mean_2C'], color='#1f77b4', linestyle='--', marker='s', linewidth=1.5, label='Melt-column mean 2C')
    ax2.plot(x, df['meltcol_mean_5C'], color='#d62728', linestyle='--', marker='s', linewidth=1.5, label='Melt-column mean 5C')
    axes[2].set_ylabel('ALT mean (m)')
    ax2.set_ylabel('Melt-column mean (m)')
    axes[2].set_title('Thermal state from nearest boreholes')
    (lines, labels1) = axes[2].get_legend_handles_labels()
    (lines2, labels2) = ax2.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels1 + labels2, ncol=2, fontsize=8, loc='upper left')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)
    fig.suptitle('Pair-wise coupling between geomorphic budget and borehole thermal state', fontsize=14)
    figsave(fig, FIG_DIR / 'FIG02_pair_budget_thermal_coupling.png')

def plot_kinematics_events(thermo_dir: Path, event_dir: Path, ice_dir: Path, event_times: pd.DataFrame) -> None:
    (fig, axes) = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    for (site_id, color) in zip(GNSS_SITES, ['#2b6cb0', '#6b7280', '#b22222']):
        ts = safe_read_csv(thermo_dir / f'GNSS_{site_id}_thermo_slump_final_timeseries.csv')
        ts['time'] = pd.to_datetime(ts['time'])
        axes[0].plot(ts['time'], pd.to_numeric(ts['w_obs_m'], errors='coerce'), color=color, linewidth=2.0, label=site_id)
    for (_, ev) in event_times[event_times['type'] == 'GNSS'].iterrows():
        tf = pd.to_datetime(ev['t_failure'], errors='coerce')
        if pd.notna(tf):
            axes[0].axvline(tf, color='#b22222', linestyle='--', linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel('Observed displacement (m)')
    axes[0].set_title('GNSS displacement trajectories')
    axes[0].legend(ncol=3, fontsize=8)
    for (site_id, color) in zip(LFJ_SITES, ['#b22222', '#2b6cb0']):
        ts = safe_read_csv(event_dir / f'LFJ_{site_id}_timeseries_clean_v5.csv')
        ts['\u8bbe\u5907\u65f6\u95f4'] = pd.to_datetime(ts['\u8bbe\u5907\u65f6\u95f4'])
        axes[1].plot(ts['\u8bbe\u5907\u65f6\u95f4'], pd.to_numeric(ts['open_mm'], errors='coerce'), color=color, linewidth=1.6, label=site_id)
    for (_, ev) in event_times[event_times['type'] == 'LFJ'].iterrows():
        tf = pd.to_datetime(ev['t_failure'], errors='coerce')
        if pd.notna(tf):
            axes[1].axvline(tf, color='#b22222', linestyle='--', linewidth=1.0, alpha=0.8)
    axes[1].set_ylabel('Opening (mm)')
    axes[1].set_title('LFJ crack-opening trajectories')
    axes[1].legend(ncol=2, fontsize=8)
    thermal_colors = {'2C': '#1f77b4', '5C': '#d62728'}
    for bh in ('2C', '5C'):
        ts = safe_read_csv(ice_dir / f'ICE_WEAKENING_timeseries_{bh}.csv')
        ts['date'] = pd.to_datetime(ts['date'])
        axes[2].plot(ts['date'], pd.to_numeric(ts['ALT_m'], errors='coerce'), color=thermal_colors[bh], linewidth=2.0, label=f'{bh} ALT')
    ax2 = axes[2].twinx()
    for bh in ('2C', '5C'):
        ts = safe_read_csv(ice_dir / f'ICE_WEAKENING_timeseries_{bh}.csv')
        ts['date'] = pd.to_datetime(ts['date'])
        ax2.plot(ts['date'], pd.to_numeric(ts['melt_frac_col_m'], errors='coerce'), color=thermal_colors[bh], linestyle='--', linewidth=1.6, label=f'{bh} melt column')
    axes[2].set_ylabel('ALT (m)')
    ax2.set_ylabel('Melt-column thickness (m)')
    axes[2].set_title('Daily thermal evolution at key boreholes')
    (lines, labels) = axes[2].get_legend_handles_labels()
    (lines2, labels2) = ax2.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, ncol=2, fontsize=8, loc='upper left')
    fig.suptitle('Integrated kinematic and thermal monitoring trajectories', fontsize=14)
    figsave(fig, FIG_DIR / 'FIG03_monitoring_kinematics_and_events.png')

def plot_site_pair_matrix(site_pair_fusion: pd.DataFrame) -> None:
    use_sites = ['7704', '7627', '9286', '3A9', '3D3']
    metrics = [('dz_point', 'Local DoD at site (m)', 'RdBu_r'), ('SCM_prob', 'SCM probability', 'viridis'), ('dist_to_patch_m', 'Distance to nearest patch (m)', 'magma_r'), ('LOD_sig_num', 'LOD significant mask', 'Greens')]
    df = site_pair_fusion[site_pair_fusion['site_id'].isin(use_sites)].copy()
    df['LOD_sig_num'] = pd.to_numeric(df['LOD_sig'], errors='coerce')
    (fig, axes) = plt.subplots(len(metrics), 1, figsize=(12, 9), constrained_layout=True)
    for (ax, (col, title, cmap)) in zip(axes, metrics):
        mat = np.full((len(use_sites), len(PAIR_ORDER)), np.nan)
        for (i, site) in enumerate(use_sites):
            for (j, pair) in enumerate(PAIR_ORDER):
                sub = df[(df['site_id'] == site) & (df['pair'] == pair)]
                if not sub.empty:
                    mat[i, j] = pd.to_numeric(sub.iloc[0][col], errors='coerce')
        if col == 'dz_point':
            vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
            im = ax.imshow(mat, aspect='auto', cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
        elif col == 'LOD_sig_num':
            im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            im = ax.imshow(mat, aspect='auto', cmap=cmap)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.ax.set_ylabel(title, rotation=90)
        ax.set_yticks(np.arange(len(use_sites)))
        ax.set_yticklabels(use_sites)
        ax.set_xticks(np.arange(len(PAIR_ORDER)))
        ax.set_xticklabels(PAIR_ORDER, rotation=25, ha='right')
        ax.set_title(title)
        for i in range(len(use_sites)):
            for j in range(len(PAIR_ORDER)):
                if np.isfinite(mat[i, j]):
                    txt = f'{mat[i, j]:.2f}' if abs(mat[i, j]) < 100 else f'{mat[i, j]:.0f}'
                    ax.text(j, i, txt, ha='center', va='center', fontsize=7, color='white' if col != 'LOD_sig_num' else 'black')
    fig.suptitle('Site-by-pair geomorphic evidence matrix', fontsize=14)
    figsave(fig, FIG_DIR / 'FIG04_site_pair_evidence_matrix.png')

def plot_7627_explanation(site_pair_fusion: pd.DataFrame, thermo_dir: Path, ice_pair: pd.DataFrame) -> None:
    (fig, axes) = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ts7627 = safe_read_csv(thermo_dir / 'GNSS_7627_thermo_slump_final_timeseries.csv')
    ts9286 = safe_read_csv(thermo_dir / 'GNSS_9286_thermo_slump_final_timeseries.csv')
    ts7627['time'] = pd.to_datetime(ts7627['time'])
    ts9286['time'] = pd.to_datetime(ts9286['time'])
    axes[0, 0].plot(ts7627['time'], pd.to_numeric(ts7627['w_obs_m'], errors='coerce'), color='#2b6cb0', linewidth=2.0, label='7627 observed')
    axes[0, 0].plot(ts9286['time'], pd.to_numeric(ts9286['w_obs_m'], errors='coerce'), color='#b22222', linewidth=2.0, label='9286 observed')
    axes[0, 0].set_ylabel('Observed displacement (m)')
    axes[0, 0].set_title('GNSS contrast: stable 7627 vs active 9286')
    axes[0, 0].legend(fontsize=8)
    sub7627 = site_pair_fusion[site_pair_fusion['site_id'] == '7627'].copy()
    sub7627['pair'] = pd.Categorical(sub7627['pair'], categories=PAIR_ORDER, ordered=True)
    sub7627 = sub7627.sort_values('pair')
    x = np.arange(sub7627.shape[0])
    axes[0, 1].bar(x - 0.18, pd.to_numeric(sub7627['dz_point'], errors='coerce'), width=0.35, color='#4c78a8', label='7627 dz_point')
    axes[0, 1].bar(x + 0.18, pd.to_numeric(sub7627['dz_mean_local'], errors='coerce'), width=0.35, color='#72b7b2', label='7627 local mean dz')
    ax12 = axes[0, 1].twinx()
    ax12.plot(x, pd.to_numeric(sub7627['SCM_prob'], errors='coerce'), color='#b22222', marker='o', linewidth=1.8, label='7627 SCM prob')
    axes[0, 1].axhline(0.0, color='0.3', linewidth=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(sub7627['pair'], rotation=25, ha='right', fontsize=8)
    axes[0, 1].set_ylabel('DoD at site (m)')
    ax12.set_ylabel('SCM probability')
    axes[0, 1].set_title('Local geomorphic response at 7627')
    (lines, labels) = axes[0, 1].get_legend_handles_labels()
    (lines2, labels2) = ax12.get_legend_handles_labels()
    axes[0, 1].legend(lines + lines2, labels + labels2, fontsize=8, loc='upper right')
    pair_2c = ice_pair[ice_pair['site_id'] == '2C'].copy()
    pair_5c = ice_pair[ice_pair['site_id'] == '5C'].copy()
    pair_2c['pair'] = pd.Categorical(pair_2c['pair'], categories=PAIR_ORDER, ordered=True)
    pair_5c['pair'] = pd.Categorical(pair_5c['pair'], categories=PAIR_ORDER, ordered=True)
    pair_2c = pair_2c.sort_values('pair')
    pair_5c = pair_5c.sort_values('pair')
    xi = np.arange(len(PAIR_ORDER))
    axes[1, 0].plot(xi, pair_2c['ALT_mean'], color='#1f77b4', marker='o', linewidth=1.8, label='2C ALT mean')
    axes[1, 0].plot(xi, pair_5c['ALT_mean'], color='#d62728', marker='o', linewidth=1.8, label='5C ALT mean')
    ax20 = axes[1, 0].twinx()
    ax20.plot(xi, pair_2c['meltcol_mean'], color='#1f77b4', linestyle='--', marker='s', linewidth=1.5, label='2C melt-column mean')
    ax20.plot(xi, pair_5c['meltcol_mean'], color='#d62728', linestyle='--', marker='s', linewidth=1.5, label='5C melt-column mean')
    axes[1, 0].set_xticks(xi)
    axes[1, 0].set_xticklabels(PAIR_ORDER, rotation=25, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('ALT mean (m)')
    ax20.set_ylabel('Melt-column mean (m)')
    axes[1, 0].set_title('Thermal weakening is stronger at 5C than at 2C')
    (lines, labels) = axes[1, 0].get_legend_handles_labels()
    (lines2, labels2) = ax20.get_legend_handles_labels()
    axes[1, 0].legend(lines + lines2, labels + labels2, fontsize=8, loc='upper left')
    explanation = 'Interpretation for the stable 7627/5C sector\n\n1. 7627 records local DoD responses, but amplitudes remain small\n   and alternate in sign across pairs.\n2. LOD-significant change occurs only once in six pairs.\n3. SCM probability stays moderate (~0.49-0.51) rather than\n   persistently high, and patch distance varies widely.\n4. 5C shows strong thaw and melt-column growth, meaning thermal\n   weakening exists, but it is not sufficient by itself.\n5. The sector lacks sustained retrogression, debris loading,\n   and shear-rate forcing, so the response is interpreted as\n   local thaw settlement / fringe adjustment, not slope-scale failure.'
    axes[1, 1].axis('off')
    axes[1, 1].text(0.02, 0.98, explanation, va='top', ha='left', fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1'))
    axes[1, 1].set_title('Professional explanation')
    fig.suptitle('Why 7627/5C shows local DoD response but remains kinematically stable', fontsize=14)
    figsave(fig, FIG_DIR / 'FIG05_7627_5C_stability_explanation.png')

def plot_site_response_space(site_summary: pd.DataFrame) -> None:
    df = site_summary[site_summary['site_id'].isin(CORE_SITES)].copy()
    (fig, ax) = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    for (_, row) in df.iterrows():
        color = ROLE_COLORS.get(row['field_role'], ROLE_COLORS['other'])
        marker = SITE_MARKERS.get(row['site_type'], 'o')
        yval = row['signal_range_m'] if row['site_type'] == 'GNSS' else row['signal_range_open_mm']
        ax.scatter(row['geomorph_scm_prob_max'], yval, s=max(60, 2500.0 / max(row['nearest_bh_distance_m'], 1.0)), marker=marker, color=color, edgecolors='black', linewidths=0.8)
        ax.text(row['geomorph_scm_prob_max'] + 0.005, yval, row['site_id'], fontsize=9)
    ax.set_xlabel('Maximum local SCM probability')
    ax.set_ylabel('Observed signal range (m for GNSS, mm for LFJ)')
    ax.set_title('Monitoring-point response space')
    ax.grid(alpha=0.25)
    figsave(fig, FIG_DIR / 'FIG06_site_response_space.png')

def main() -> None:
    plt.style.use('seaborn-v0_8-whitegrid')
    features = safe_read_csv(OUT_DIR / 'PATCH' / 'BH_GNSS_geomorph_features.csv')
    geomorph_budget = safe_read_csv(OUT_DIR / 'GEOMORPH_CHANGE_13' / 'STATS' / 'geomorph_budget_all.csv')
    ice_pair = safe_read_csv(OUT_DIR / 'PINN_THERMAL_10A' / 'ICE_WEAKENING' / 'ICE_WEAKENING_pair_aligned.csv')
    event_times = safe_read_csv(OUT_DIR / 'EVENT_OUT' / 'event_times_v5.csv')
    stability = safe_read_csv(OUT_DIR / 'EVENT_OUT' / 'stability_metrics_v5.csv')
    with open(OUT_DIR / '10B_THERMO_SLUMP_FINAL' / 'GNSS_to_BH_nearest_mapping.json', 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    site_catalog = build_site_catalog(features)
    site_catalog = attach_roles(site_catalog, stability)
    site_catalog = compute_nearest_bh(site_catalog)
    site_catalog = override_gnss_mapping(site_catalog, mapping)
    pair_rates = {}
    thermo_dir = OUT_DIR / '10B_THERMO_SLUMP_FINAL'
    for site_id in GNSS_SITES:
        p = thermo_dir / f'GNSS_{site_id}_pair_rates.csv'
        if p.exists():
            pair_rates[site_id] = safe_read_csv(p)
    site_pair_fusion = build_site_pair_fusion(site_catalog, features, ice_pair, pair_rates)
    site_summary = build_site_summary(site_catalog, features, event_times, stability, thermo_dir, OUT_DIR / 'EVENT_OUT')
    pair_summary = build_pair_summary(geomorph_budget, ice_pair, event_times, site_pair_fusion)
    site_catalog.sort_values(['site_type', 'site_id']).to_csv(TABLE_DIR / 'site_catalog.csv', index=False, encoding='utf-8-sig')
    site_pair_fusion.to_csv(TABLE_DIR / 'site_pair_multisource_fusion.csv', index=False, encoding='utf-8-sig')
    site_summary.to_csv(TABLE_DIR / 'site_multisource_summary.csv', index=False, encoding='utf-8-sig')
    pair_summary.to_csv(TABLE_DIR / 'pair_multisource_summary.csv', index=False, encoding='utf-8-sig')
    scm_rasters = discover_raster(OUT_DIR / 'SCM_PROB', 'SCM_prob_*.tif')
    dod_rasters = discover_raster(BASE_DIR / 'data' / 'dod_0p1m', '*.tif')
    latest_pair = '251017-230924'
    export_latest_rasters(site_catalog[site_catalog['site_id'].isin(BH_SITES + CORE_SITES + ['7607', '7679'])], scm_rasters[latest_pair], dod_rasters[latest_pair])
    print(f'Paper-result tables and raster payload written to: {PAPER_DIR}')
if __name__ == '__main__':
    main()
