from __future__ import annotations
import fnmatch
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
REQUIRED_TOP_LEVEL_KEYS = {'authoritative_data_sources', 'deprecated_sources', 'output_roots', 'filename_patterns', 'validation_rules'}

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _default_manifest_path() -> Path:
    return _project_root() / 'DATA_MANIFEST.yaml'

def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding='utf-8')
    try:
        import yaml
        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f'Manifest must parse to a mapping: {path}')
    return data

def load_manifest(manifest_path: str | Path | None=None) -> dict[str, Any]:
    """Load DATA_MANIFEST.yaml and validate its required top-level sections."""
    path = Path(manifest_path) if manifest_path is not None else _default_manifest_path()
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f'DATA_MANIFEST.yaml not found: {path}')
    manifest = _load_yaml_or_json(path)
    missing = REQUIRED_TOP_LEVEL_KEYS.difference(manifest)
    if missing:
        raise KeyError(f'DATA_MANIFEST.yaml is missing required keys: {sorted(missing)}')
    manifest['_manifest_path'] = str(path)
    return manifest

def _source_entry(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    sources = manifest.get('authoritative_data_sources', {})
    if key not in sources:
        raise KeyError(f'Unknown authoritative data source key: {key}')
    entry = sources[key]
    if isinstance(entry, str):
        entry = {'path': entry}
    if not isinstance(entry, dict) or 'path' not in entry:
        raise ValueError(f'Invalid data source entry for {key!r}; expected a path.')
    return entry

def _normalize_for_compare(path: str | Path) -> str:
    raw = str(path)
    expanded = os.path.expandvars(os.path.expanduser(raw))
    try:
        normalized = str(Path(expanded).resolve())
    except OSError:
        normalized = str(Path(expanded).absolute())
    return os.path.normcase(os.path.normpath(normalized))

def _pattern_to_absolute(pattern: str | Path) -> str:
    raw = str(pattern)
    expanded = os.path.expandvars(os.path.expanduser(raw))
    if not re.match('^[A-Za-z]:[\\\\/]', expanded) and (not expanded.startswith('\\\\')):
        expanded = str((_project_root() / expanded).absolute())
    return os.path.normcase(os.path.normpath(expanded))

def assert_not_deprecated(path: str | Path, manifest: dict[str, Any] | None=None) -> None:
    """Raise if path is equal to, below, or matched by any deprecated source."""
    manifest = manifest or load_manifest()
    target = _normalize_for_compare(path)
    for (key, entry) in manifest.get('deprecated_sources', {}).items():
        if isinstance(entry, str):
            entry = {'path': entry}
        reason = entry.get('reason', 'Deprecated source.')
        dep_path = entry.get('path')
        dep_glob = entry.get('glob')
        if dep_path:
            deprecated = _normalize_for_compare(dep_path)
            if target == deprecated or target.startswith(deprecated + os.sep):
                raise ValueError(f'Path is deprecated by {key}: {path} ({reason})')
        if dep_glob:
            pattern = _pattern_to_absolute(dep_glob)
            if fnmatch.fnmatch(target, pattern):
                raise ValueError(f'Path matches deprecated pattern {key}: {path} ({reason})')

def get_data_source(key: str, manifest: dict[str, Any] | None=None, must_exist: bool | None=None) -> Path:
    """Return an authoritative data source Path after manifest and deprecated-source checks."""
    manifest = manifest or load_manifest()
    entry = _source_entry(manifest, key)
    path = Path(entry['path'])
    if not path.is_absolute():
        path = _project_root() / path
    path = path.resolve()
    assert_not_deprecated(path, manifest=manifest)
    if must_exist is None:
        must_exist = key in set(manifest.get('validation_rules', {}).get('must_exist', []))
    if must_exist and (not path.exists()):
        raise FileNotFoundError(f'Required data source does not exist for key {key!r}: {path}')
    return path

def _sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    date_tokens = re.findall('(?<!\\d)(\\d{6})(?!\\d)', name)
    if date_tokens:
        return (int(date_tokens[-1]), name)
    return (int(path.stat().st_mtime), name)

def list_latest_files(source_key: str, pattern_key: str | None=None, count: int | None=None, manifest: dict[str, Any] | None=None) -> list[Path]:
    """List files from a manifest source using a manifest filename pattern without reading file contents."""
    manifest = manifest or load_manifest()
    root = get_data_source(source_key, manifest=manifest)
    pattern = '*'
    if pattern_key is not None:
        patterns = manifest.get('filename_patterns', {})
        pattern = patterns.get(pattern_key, pattern_key)
    files = sorted((p for p in root.glob(pattern) if p.is_file()), key=_sort_key)
    for path in files:
        assert_not_deprecated(path, manifest=manifest)
    if count is not None:
        files = files[-count:]
    return files

def write_input_audit(task_name: str, sources: list[str] | dict[str, str | list[str]], output_dir: str | Path, manifest: dict[str, Any] | None=None) -> Path:
    """Write an input_audit.md file for a task before heavy reads or computation."""
    manifest = manifest or load_manifest()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / 'input_audit.md'
    if isinstance(sources, list):
        source_items = {key: key for key in sources}
    else:
        source_items = sources
    lines = ['# Input Audit', '', f'- Task: `{task_name}`', f"- Timestamp: `{datetime.now().isoformat(timespec='seconds')}`", f"- Manifest: `{manifest.get('_manifest_path', _default_manifest_path())}`", '', '## Sources', '']
    for (source_key, pattern_keys) in source_items.items():
        root = get_data_source(source_key, manifest=manifest)
        lines.append(f'### {source_key}')
        lines.append('')
        lines.append(f'- Root: `{root}`')
        keys = pattern_keys if isinstance(pattern_keys, list) else [pattern_keys]
        for pattern_key in keys:
            if pattern_key == source_key:
                lines.append('- Pattern: `(not specified)`')
                lines.append('- Matched files: `(source root only)`')
                continue
            files = list_latest_files(source_key, pattern_key, manifest=manifest)
            lines.append(f"- Pattern `{pattern_key}`: `{manifest['filename_patterns'].get(pattern_key, pattern_key)}`")
            if files:
                for path in files:
                    lines.append(f'  - `{path.name}`')
            else:
                lines.append('  - `(no files matched)`')
        lines.append('')
    lines.extend(['## Validation', '', '- Manifest loaded successfully.', '- Deprecated-source checks passed for listed roots and files.', '- No large raster, LAS, or model payload was read by this audit helper.', ''])
    audit_path.write_text('\n'.join(lines), encoding='utf-8')
    return audit_path
