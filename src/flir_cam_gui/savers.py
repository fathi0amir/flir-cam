"""Image persistence: raw binary + JSON sidecar, and HDF5 writers.

Pure-logic helpers (no Qt, no PySpin) so they can be unit-tested and reused.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

APP_NAME = "flir-cam-gui"

#: Sidecar schema version.  2 moved the per-frame metadata out of the JSON
#: (where a few thousand frames made it tens of MB) into a sibling Parquet
#: table; the JSON keeps only global attributes plus a reference to it.
SIDECAR_SCHEMA = 2


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------

def build_sidecar(
    *,
    image_info: Dict[str, Any],
    camera_settings: Dict[str, Any],
    display_settings: Optional[Dict[str, Any]] = None,
    frame_table: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the JSON sidecar dictionary saved next to every image file.

    The sidecar holds *global* attributes only; per-frame acquisition metadata
    lives in the Parquet table described by ``frame_table`` so recording a few
    thousand frames does not turn the JSON into a multi-megabyte file.

    Parameters
    ----------
    image_info : details about the pixel payload (name, width, height, dtype,
        byte count, pixel format string, storage description).
    camera_settings : full GenICam node-map snapshot (as produced by the
        camera worker).
    display_settings : optional live-view display settings that were active.
    frame_table : reference to the per-frame Parquet table, as returned by
        :func:`frame_table_entry`.
    """
    sidecar: Dict[str, Any] = {
        "schema": SIDECAR_SCHEMA,
        "application": APP_NAME,
        "app_version": _app_version(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file": image_info,
        "camera_settings": camera_settings,
    }
    if frame_table is not None:
        sidecar["frame_metadata"] = dict(frame_table)
    if display_settings:
        sidecar["display_settings"] = display_settings
    return sidecar


def write_sidecar(path: Path, sidecar: Dict[str, Any]) -> None:
    path.write_text(json.dumps(sidecar, indent=2, default=_json_default), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-frame metadata table (Parquet)
# ---------------------------------------------------------------------------

FRAME_TABLE_SUFFIX = ".frames.parquet"
FRAME_TABLE_FORMAT = "parquet"

#: Per-frame keys that are identical for every frame of a recording (or that
#: already live in the sidecar's global ``file`` / ``camera_settings``), so they
#: are not duplicated in the frame table.  ``chunk`` is dropped as a whole: its
#: exposure/gain are camera settings and its frame id/timestamp are already
#: covered by the remaining ``frame_id`` and ``device_timestamp_ns`` columns.
GLOBAL_FRAME_KEYS = frozenset({
    "camera_model", "sensor", "bit_depth", "width", "height",
    "pixel_format", "acquired_shape", "acquired_dtype", "chunk",
})


def frame_table_path(sidecar_path: Path) -> Path:
    """Where the frame table lives for a sidecar: ``x.h5.json`` ->
    ``x.h5.frames.parquet`` (``sequence.json`` -> ``sequence.frames.parquet``)."""
    sidecar_path = Path(sidecar_path)
    name = sidecar_path.name
    stem = name[:-len(".json")] if name.endswith(".json") else name
    return sidecar_path.with_name(stem + FRAME_TABLE_SUFFIX)


def frame_table_entry(path: Path, rows: int) -> Dict[str, Any]:
    """Sidecar reference to a written frame table (name only, next to it)."""
    return {"format": FRAME_TABLE_FORMAT, "file": Path(path).name,
            "rows": int(rows)}


def _normalise_frames(frames: Any) -> List[Dict[str, Any]]:
    """Accept one frame dict, a list of them, or ``{"frames": [...]}``."""
    if isinstance(frames, dict):
        rows = frames.get("frames")
        frames = rows if isinstance(rows, list) else [frames]
    return [dict(f or {}) for f in frames or []]


def write_frame_table(path: Path, frames: Sequence[Dict[str, Any]]) -> Path:
    """Write per-frame metadata as a Parquet table, one row per frame.

    Rows are normalised through JSON first so NumPy scalars and other values
    the acquisition code hands over become plain Python types; that also keeps
    the inferred Arrow schema stable across frames.  Columns in
    :data:`GLOBAL_FRAME_KEYS` are dropped: they are the same for every frame
    (geometry, sensor, ``chunk``) and already live in the sidecar's global
    ``file`` / ``camera_settings``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: List[Dict[str, Any]] = []
    for index, frame in enumerate(_normalise_frames(frames)):
        clean = json.loads(json.dumps(frame, default=_json_default))
        clean = {k: v for k, v in clean.items() if k not in GLOBAL_FRAME_KEYS}
        rows.append({"frame_index": index, **clean})
    if rows:
        table = pa.Table.from_pylist(rows)
        order = ["frame_index"] + [c for c in table.column_names if c != "frame_index"]
        table = table.select(order)
    else:
        table = pa.table({"frame_index": pa.array([], type=pa.int64())})
    path = Path(path)
    pq.write_table(table, path, compression="snappy")
    return path


def read_frame_table(path: Path) -> List[Dict[str, Any]]:
    """Read a frame table back into a list of per-frame dicts."""
    import pyarrow.parquet as pq

    return [dict(row) for row in pq.read_table(Path(path)).to_pylist()]


def write_frame_metadata(sidecar_path: Path,
                         frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Write the frame table beside a sidecar and return its reference."""
    rows = _normalise_frames(frames)
    table = write_frame_table(frame_table_path(sidecar_path), rows)
    return frame_table_entry(table, len(rows))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bytes):
        return {"__bytes__": len(obj)}
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _app_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Raw binary
# ---------------------------------------------------------------------------

def write_raw(path: Path, arr: np.ndarray) -> None:
    """Write raw pixel data: row-major, C-contiguous, little-endian bytes."""
    Path(path).write_bytes(np.ascontiguousarray(arr).tobytes(order="C"))


def frame_geometry(arr: np.ndarray) -> Tuple[int, int, int]:
    """(height, width, channels) of one frame.

    A third axis is only a channel axis for colour frames -- using ``shape[-1]``
    blindly reports width=3 for an RGB image.
    """
    if arr.ndim == 3:
        return int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1]), 1
    return 1, int(arr.shape[-1]), 1


def raw_image_info(path: Path, arr: np.ndarray, pixel_format: str) -> Dict[str, Any]:
    """Describe raw pixel payload. For colour frames the channel axis is kept
    separate so (height, width) never collapses into the channel count."""
    height, width, channels = frame_geometry(arr)
    return {
        "name": Path(path).name,
        "type": "raw_binary",
        "description": (
            "Raw pixel data, row-major (C order), no header. "
            "Reshape as (height, width[, channels]) in this dtype to recover "
            "the image."
        ),
        "byte_order": "little",
        "width": width,
        "height": height,
        "channels": channels,
        "shape": [int(s) for s in arr.shape],
        "dtype": str(arr.dtype),
        "bytes": int(arr.nbytes),
        "pixel_format": pixel_format,
    }


def save_raw_with_sidecar(
    path: Path,
    arr: np.ndarray,
    *,
    pixel_format: str,
    frame_meta: Dict[str, Any],
    camera_settings: Dict[str, Any],
    display_settings: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save raw pixel binary + a sidecar holding every setting for that image."""
    path = Path(path)
    write_raw(path, arr)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    table = write_frame_metadata(sidecar_path, [frame_meta])
    write_sidecar(
        sidecar_path,
        build_sidecar(
            image_info=raw_image_info(path, arr, pixel_format),
            camera_settings=camera_settings,
            display_settings=display_settings,
            frame_table=table,
        ),
    )
    return path


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------

def save_hdf5(
    path: Path,
    arr: np.ndarray,
    *,
    pixel_format: str,
    frame_meta: Dict[str, Any],
    camera_settings: Dict[str, Any],
    display_settings: Optional[Dict[str, Any]] = None,
    include_sidecar: bool = True,
) -> Path:
    """Save a single frame as HDF5 ('/image' dataset + metadata attributes)."""
    import h5py

    path = Path(path)
    with h5py.File(path, "w") as f:
        f.create_dataset("image", data=arr)
        grp = f.require_group("metadata")
        grp.attrs["pixel_format"] = pixel_format
        grp.attrs["frame_metadata_json"] = json.dumps(frame_meta or {}, default=_json_default)
        grp.attrs["camera_settings_json"] = json.dumps(camera_settings, default=_json_default)
        if display_settings:
            grp.attrs["display_settings_json"] = json.dumps(display_settings, default=_json_default)
        f.attrs["application"] = APP_NAME
        f.attrs["saved_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if camera_settings:
            for key, value in flatten_settings(camera_settings).items():
                f.attrs[f"cam/{key}"] = value
    if include_sidecar:
        sidecar_path = path.with_suffix(path.suffix + ".json")
        table = write_frame_metadata(sidecar_path, [frame_meta])
        write_sidecar(
            sidecar_path,
            build_sidecar(
                image_info={
                    "name": path.name, "type": "hdf5_single",
                    "width": frame_geometry(arr)[1],
                    "height": frame_geometry(arr)[0],
                    "channels": frame_geometry(arr)[2],
                    "dtype": str(arr.dtype), "pixel_format": pixel_format},
                camera_settings=camera_settings,
                display_settings=display_settings, frame_table=table),
        )
    return path


def save_hdf5_sequence(
    path: Path,
    frames: Sequence[np.ndarray],
    metas: Sequence[Dict[str, Any]],
    *,
    pixel_format: str,
    camera_settings: Dict[str, Any],
    display_settings: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a recorded burst as one HDF5 dataset ('/images', N x H x W[ x C])."""
    import h5py

    if not frames:
        raise ValueError("no frames captured")
    path = Path(path)
    frame_meta_list: List[Dict[str, Any]] = [dict(m or {}) for m in metas]
    with h5py.File(path, "w") as f:
        arrs = [np.ascontiguousarray(a) for a in frames]
        # Frames may differ in size (ROI changes mid-record). Pad stays: refuse
        # mismatched shapes to avoid silently corrupting the dataset.
        shapes = {a.shape for a in arrs}
        if len(shapes) != 1:
            raise ValueError(f"mixed frame shapes {shapes}; cannot stack sequence")
        f.create_dataset(
            "images",
            data=np.stack(arrs),
            chunks=(1,) + arrs[0].shape,
            compression="lzf",
        )
        grp = f.require_group("metadata")
        grp.attrs["pixel_format"] = pixel_format
        grp.attrs["frame_count"] = len(frame_meta_list)
        grp.attrs["camera_settings_json"] = json.dumps(camera_settings, default=_json_default)
        if display_settings:
            grp.attrs["display_settings_json"] = json.dumps(display_settings, default=_json_default)
        f.attrs["application"] = APP_NAME
        f.attrs["saved_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if camera_settings:
            for key, value in flatten_settings(camera_settings).items():
                f.attrs[f"cam/{key}"] = value
    # Per-frame metadata goes to a columnar table beside the JSON sidecar.
    sidecar_path = path.with_suffix(path.suffix + ".json")
    table = write_frame_metadata(sidecar_path, frame_meta_list)
    write_sidecar(sidecar_path, build_sidecar(
        image_info={
            "name": path.name,
            "type": "hdf5_sequence",
            "width": frame_geometry(frames[0])[1],
            "height": frame_geometry(frames[0])[0],
            "channels": frame_geometry(frames[0])[2],
            "dtype": str(frames[0].dtype),
            "frames": len(frames),
            "pixel_format": pixel_format,
        },
        camera_settings=camera_settings,
        display_settings=display_settings,
        frame_table=table,
    ))
    return path


# ---------------------------------------------------------------------------
# NumPy and BigTIFF
# ---------------------------------------------------------------------------

def _save_array_with_sidecar(path: Path, arr: np.ndarray, *, kind: str,
                             pixel_format: str, frame_meta: Any,
                             camera_settings: Dict[str, Any],
                             display_settings: Optional[Dict[str, Any]],
                             frames: Optional[int] = None) -> Path:
    path = Path(path)
    image = arr[0] if frames is not None and arr.ndim >= 3 else arr
    height, width, channels = frame_geometry(image)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    table = write_frame_metadata(sidecar_path, _normalise_frames(frame_meta))
    write_sidecar(sidecar_path, build_sidecar(
        image_info={"name": path.name, "type": kind, "width": width,
                    "height": height, "channels": channels,
                    "shape": list(arr.shape), "dtype": str(arr.dtype),
                    "bytes": int(arr.nbytes), "pixel_format": pixel_format,
                    **({"frames": int(frames)} if frames is not None else {})},
        camera_settings=camera_settings,
        display_settings=display_settings, frame_table=table))
    return path


def save_npy(path: Path, arr: np.ndarray, *, pixel_format: str,
             frame_meta: Dict[str, Any], camera_settings: Dict[str, Any],
             display_settings: Optional[Dict[str, Any]] = None,
             include_sidecar: bool = True) -> Path:
    """Save an uncompressed NumPy array, preserving its shape and dtype."""
    path = Path(path)
    with path.open("wb") as f:
        np.save(f, np.asarray(arr), allow_pickle=False)
    if include_sidecar:
        _save_array_with_sidecar(path, arr, kind="npy", pixel_format=pixel_format,
                                 frame_meta=frame_meta, camera_settings=camera_settings,
                                 display_settings=display_settings)
    return path


def save_npy_sequence(path: Path, frames: Sequence[np.ndarray],
                      metas: Sequence[Dict[str, Any]], *, pixel_format: str,
                      camera_settings: Dict[str, Any],
                      display_settings: Optional[Dict[str, Any]] = None) -> Path:
    if not frames:
        raise ValueError("no frames captured")
    arrays = [np.ascontiguousarray(a) for a in frames]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("mixed frame shapes; cannot save a NumPy stack")
    stack = np.stack(arrays)
    path = Path(path)
    with path.open("wb") as f:
        np.save(f, stack, allow_pickle=False)
    return _save_array_with_sidecar(path, stack, kind="npy_sequence",
                                    pixel_format=pixel_format,
                                    frame_meta={"frames": list(metas)},
                                    camera_settings=camera_settings,
                                    display_settings=display_settings,
                                    frames=len(arrays))


def _write_bigtiff(path: Path, arr: np.ndarray) -> None:
    import tifffile
    # Explicit BigTIFF is intentional: classic TIFF is limited to 4 GiB.
    tifffile.imwrite(path, np.asarray(arr), bigtiff=True, photometric="minisblack" if arr.ndim < 3 else None)


def save_tiff(path: Path, arr: np.ndarray, *, pixel_format: str,
              frame_meta: Dict[str, Any], camera_settings: Dict[str, Any],
              display_settings: Optional[Dict[str, Any]] = None,
              include_sidecar: bool = True) -> Path:
    """Save one image as BigTIFF (never classic TIFF)."""
    path = Path(path)
    _write_bigtiff(path, arr)
    if include_sidecar:
        _save_array_with_sidecar(path, arr, kind="bigtiff", pixel_format=pixel_format,
                                 frame_meta=frame_meta, camera_settings=camera_settings,
                                 display_settings=display_settings)
    return path


def save_tiff_sequence(path: Path, frames: Sequence[np.ndarray],
                       metas: Sequence[Dict[str, Any]], *, pixel_format: str,
                       camera_settings: Dict[str, Any],
                       display_settings: Optional[Dict[str, Any]] = None) -> Path:
    if not frames:
        raise ValueError("no frames captured")
    arrays = [np.ascontiguousarray(a) for a in frames]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("mixed frame shapes; cannot save a BigTIFF stack")
    stack = np.stack(arrays)
    path = Path(path)
    _write_bigtiff(path, stack)
    return _save_array_with_sidecar(path, stack, kind="bigtiff_sequence",
                                    pixel_format=pixel_format,
                                    frame_meta={"frames": list(metas)},
                                    camera_settings=camera_settings,
                                    display_settings=display_settings,
                                    frames=len(arrays))


# ---------------------------------------------------------------------------
# Zarr + Blosc (lz4/zstd, level 1-3, shuffle)
# ---------------------------------------------------------------------------

BLOSC_CNAMES = ("lz4", "zstd")
BLOSC_CLEVELS = (1, 2, 3)
BLOSC_SHUFFLES = ("noshuffle", "shuffle", "bitshuffle")


def blosc_codec(
    cname: str = "lz4", clevel: int = 1, shuffle: str = "shuffle"
):
    """Build a Zarr v3 Blosc codec; specification allows level 1-3 + shuffle."""
    if cname not in BLOSC_CNAMES:
        raise ValueError(f"blosc cname must be one of {BLOSC_CNAMES}, got {cname!r}")
    clevel = int(clevel)
    if clevel not in BLOSC_CLEVELS:
        raise ValueError(f"blosc clevel must be one of {BLOSC_CLEVELS}, got {clevel}")
    if shuffle not in BLOSC_SHUFFLES:
        raise ValueError(f"blosc shuffle must be one of {BLOSC_SHUFFLES}, got {shuffle!r}")
    import zarr.codecs

    return zarr.codecs.BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle)


def zarr_store_meta(store_path: Path, array_name: str = "image") -> Dict[str, Any]:
    """Read a Zarr store's array metadata (codecs etc.) for verification."""
    meta_path = Path(store_path) / array_name / "zarr.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _zarr_attrs(group, *, pixel_format, camera_settings,
                display_settings, frame_meta, array=None):
    attrs = {
        "pixel_format": pixel_format,
        "application": APP_NAME,
        "camera_settings_json": json.dumps(camera_settings or {},
                                           default=_json_default),
    }
    if frame_meta is not None:
        attrs["frame_metadata_json"] = json.dumps(frame_meta, default=_json_default)
    if display_settings:
        attrs["display_settings_json"] = json.dumps(display_settings,
                                                    default=_json_default)
    group.attrs.update(attrs)
    if array is not None:
        array.attrs.update(attrs)


def save_zarr(
    path: Path,
    arr: np.ndarray,
    *,
    pixel_format: str,
    frame_meta: Dict[str, Any],
    camera_settings: Dict[str, Any],
    display_settings: Optional[Dict[str, Any]] = None,
    blosc_cname: str = "lz4",
    blosc_clevel: int = 1,
    blosc_shuffle: str = "shuffle",
    include_sidecar: bool = True,
) -> Path:
    """Save a single frame as a Zarr v3 store with Blosc-compressed chunks."""
    import zarr
    from zarr.storage import LocalStore

    path = Path(path)
    codec = blosc_codec(blosc_cname, blosc_clevel, blosc_shuffle)
    if path.suffix != ".zarr":
        path = path.with_suffix(".zarr")
    group = zarr.open_group(store=LocalStore(root=path), mode="w")
    z = group.create_array(
        "image", shape=arr.shape, dtype=arr.dtype,
        chunks=arr.shape, compressors=[codec], overwrite=True,
    )
    z[...] = arr
    _zarr_attrs(group, pixel_format=pixel_format, camera_settings=camera_settings,
                display_settings=display_settings, frame_meta=frame_meta, array=z)
    if include_sidecar:
        sidecar_path = path.with_suffix(path.suffix + ".json")
        table = write_frame_metadata(sidecar_path, [frame_meta])
        write_sidecar(
            sidecar_path,
            build_sidecar(
                image_info={
                    "name": path.name, "type": "zarr_single", "format": "zarr3",
                    "width": frame_geometry(arr)[1],
                    "height": frame_geometry(arr)[0],
                    "channels": frame_geometry(arr)[2],
                    "dtype": str(arr.dtype), "pixel_format": pixel_format,
                    "compressor": {"name": "blosc", "cname": blosc_cname,
                                   "clevel": int(blosc_clevel),
                                   "shuffle": blosc_shuffle},
                },
                camera_settings=camera_settings,
                display_settings=display_settings, frame_table=table),
        )
    return path


def save_zarr_sequence(
    path: Path,
    frames: Sequence[np.ndarray],
    metas: Sequence[Dict[str, Any]],
    *,
    pixel_format: str,
    camera_settings: Dict[str, Any],
    display_settings: Optional[Dict[str, Any]] = None,
    blosc_cname: str = "lz4",
    blosc_clevel: int = 1,
    blosc_shuffle: str = "shuffle",
) -> Path:
    """Record a burst as one Zarr array ('/image', frames along axis 0)."""
    import zarr
    from zarr.storage import LocalStore

    if not frames:
        raise ValueError("no frames captured")
    path = Path(path)
    if path.suffix != ".zarr":
        path = path.with_suffix(".zarr")
    arrs = [np.ascontiguousarray(a) for a in frames]
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"mixed frame shapes {shapes}; cannot stack sequence")
    shape = arrs[0].shape
    codec = blosc_codec(blosc_cname, blosc_clevel, blosc_shuffle)
    group = zarr.open_group(store=LocalStore(root=path), mode="w")
    z = group.create_array(
        "image", shape=(len(arrs), *shape), dtype=arrs[0].dtype,
        chunks=(1, *shape), compressors=[codec], overwrite=True,
    )
    z[...] = np.stack(arrs)
    # Per-frame metadata is *not* stored in the store's JSON attributes any
    # more: a few thousand frames turned zarr.json into megabytes of JSON.
    _zarr_attrs(group, pixel_format=pixel_format, camera_settings=camera_settings,
                display_settings=display_settings, frame_meta=None, array=z)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    table = write_frame_metadata(sidecar_path, [dict(m or {}) for m in metas])
    write_sidecar(
        sidecar_path,
        build_sidecar(
            image_info={
                "name": path.name, "type": "zarr_sequence", "format": "zarr3",
                "width": frame_geometry(arrs[0])[1],
                "height": frame_geometry(arrs[0])[0],
                "channels": frame_geometry(arrs[0])[2],
                "dtype": str(arrs[0].dtype), "frames": len(arrs),
                "pixel_format": pixel_format,
                "compressor": {"name": "blosc", "cname": blosc_cname,
                               "clevel": int(blosc_clevel),
                               "shuffle": blosc_shuffle},
            },
            camera_settings=camera_settings, display_settings=display_settings,
            frame_table=table),
    )
    return path


def flatten_settings(tree: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Flatten the nested settings snapshot into 'Category/Feature: value'."""
    flat: Dict[str, str] = {}
    for name, node in tree.items():
        if not isinstance(node, dict):
            continue
        if node.get("__type__") == "category":
            flat.update(flatten_settings(node.get("features", {}), f"{prefix}{name}/"))
        elif node.get("readable"):
            flat[f"{prefix}{name}"] = str(node.get("value", ""))
    return flat
