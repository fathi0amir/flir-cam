import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from flir_cam_gui import savers


def meta_of(arr):
    return {"width": arr.shape[1], "height": arr.shape[0],
            "pixel_format": "Mono16", "frame_id": 7,
            "acquired_shape": list(arr.shape), "acquired_dtype": str(arr.dtype)}


SETTINGS = {
    "camera": {"__type__": "category", "display": "Root", "writable": False, "features": {
        "AcquisitionControl": {"__type__": "category", "display": "Acquisition Control",
                               "writable": False, "features": {
                                   "ExposureTime": {"kind": "float", "value": 12000.0,
                                                    "readable": True, "writable": True}}},
        "DeviceControl": {"__type__": "category", "display": "Device Control",
                          "writable": False, "features": {
                              "DeviceModelName": {"kind": "string", "value": "TestCam",
                                                  "readable": True, "writable": False}}}}},
    "stream": {},
}

DISPLAY = {"palette": "grayscale", "auto_levels": True, "bit_depth": 12}


def test_raw_and_sidecar(tmp_path: Path):
    arr = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8)
    p = tmp_path / "frame.raw"
    out = savers.save_raw_with_sidecar(
        p, arr, pixel_format="Mono16", frame_meta=meta_of(arr),
        camera_settings=SETTINGS, display_settings=DISPLAY)
    raw = out.read_bytes()
    assert len(raw) == arr.nbytes
    back = np.frombuffer(raw, dtype=arr.dtype).reshape(6, 8)
    assert np.array_equal(back, arr)
    sidecar = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    assert sidecar["file"]["dtype"] == "uint16"
    assert sidecar["file"]["width"] == 8 and sidecar["file"]["height"] == 6
    assert sidecar["file"]["bytes"] == arr.nbytes
    assert sidecar["schema"] == 2
    assert sidecar["frame_metadata"]["format"] == "parquet"
    assert sidecar["frame_metadata"]["rows"] == 1
    rows = savers.read_frame_table(Path(str(out) + ".frames.parquet"))
    assert rows[0]["frame_id"] == 7
    assert "saved_at_utc" in sidecar
    assert sidecar["display_settings"]["palette"] == "grayscale"
    flat = savers.flatten_settings(SETTINGS)
    assert flat["camera/AcquisitionControl/ExposureTime"] == "12000.0"
    assert sidecar["camera_settings"]["camera"]["features"]["DeviceControl"]["features"][
               "DeviceModelName"]["value"] == "TestCam"


def test_hdf5_single(tmp_path: Path):
    arr = np.zeros((4, 5), dtype=np.uint16)
    arr[1, 1] = 999
    p = tmp_path / "image.h5"
    out = savers.save_hdf5(p, arr, pixel_format="Mono16",
                           frame_meta=meta_of(arr), camera_settings=SETTINGS,
                           display_settings=DISPLAY)
    with h5py.File(out, "r") as f:
        assert np.array_equal(f["image"][:], arr)
        assert f["metadata"].attrs["pixel_format"] == "Mono16"
        assert f["metadata"].attrs["frame_metadata_json"]  # json string present
        assert 'AcquisitionControl/ExposureTime' in set(f.attrs.keys()) or any(
            k.startswith('cam/') for k in f.attrs.keys())
        # camera settings flattened attribute values
        cam_attrs = {k: v for k, v in f.attrs.items() if k.startswith("cam/")}
        assert "cam/camera/AcquisitionControl/ExposureTime" in cam_attrs
        assert "12000.0" in str(cam_attrs["cam/camera/AcquisitionControl/ExposureTime"])
    sc = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    assert sc["file"]["type"] == "hdf5_single"


def test_hdf5_sequence(tmp_path: Path):
    meta = {"pixel_format": "Mono16", "frame_id": 0}
    frames = [np.full((4, 5), i, dtype=np.uint16) for i in range(3)]
    p = tmp_path / "seq.hdf5"
    out = savers.save_hdf5_sequence(p, frames, [dict(meta, frame_id=i) for i in range(3)],
                                    pixel_format="Mono16", camera_settings=SETTINGS,
                                    display_settings=DISPLAY)
    with h5py.File(out, "r") as f:
        dset = f["images"]
        assert dset.shape == (3, 4, 5)
        assert list(dset[2].ravel()) == [2] * 20
        assert f["metadata"].attrs["frame_count"] == 3
    sc_path = out.with_suffix(out.suffix + ".json")
    sidecar = json.loads(sc_path.read_text(encoding="utf-8"))
    assert sidecar["file"]["frames"] == 3
    assert sidecar["frame_metadata"] == {"format": "parquet",
                                         "file": "seq.hdf5.frames.parquet",
                                         "rows": 3}
    rows = savers.read_frame_table(out.with_name("seq.hdf5.frames.parquet"))
    assert [r["frame_index"] for r in rows] == [0, 1, 2]
    assert rows[1]["frame_id"] == 1


def test_sequence_rejects_mixed_shapes(tmp_path):
    frames = [np.zeros((4, 5), dtype=np.uint16), np.zeros((6, 7), dtype=np.uint16)]
    with pytest.raises(ValueError):
        savers.save_hdf5_sequence(tmp_path / "s.h5", frames, [{}, {}],
                                  pixel_format="Mono16", camera_settings=SETTINGS)


# --- Zarr + Blosc -----------------------------------------------------------

BLOSC = {"blosc_cname": "zstd", "blosc_clevel": 3, "blosc_shuffle": "shuffle"}


def test_zarr_single(tmp_path: Path):
    arr = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8)
    out = savers.save_zarr(tmp_path / "frame", arr, pixel_format="Mono16",
                           frame_meta=meta_of(arr), camera_settings=SETTINGS,
                           display_settings=DISPLAY, **BLOSC)
    import zarr
    from zarr.storage import LocalStore
    g = zarr.open_group(store=LocalStore(root=out), mode="r")
    assert np.array_equal(g["image"][:], arr)
    meta = savers.zarr_store_meta(out)
    codec = [c for c in meta["codecs"] if c["name"] == "blosc"][0]
    assert codec["name"] == "blosc"
    assert codec["configuration"]["cname"] == "zstd"
    assert codec["configuration"]["clevel"] == 3
    assert codec["configuration"]["shuffle"] == "shuffle"
    assert meta["attributes"]["pixel_format"] == "Mono16"
    cam_json = meta["attributes"]["camera_settings_json"]
    assert "AcquisitionControl" in cam_json and "ExposureTime" in cam_json and "12000.0" in cam_json
    side = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    assert side["file"]["type"] == "zarr_single"
    assert side["file"]["compressor"] == {"name": "blosc", "cname": "zstd",
                                          "clevel": 3, "shuffle": "shuffle"}
    table = savers.frame_table_path(Path(str(out) + ".json"))
    assert savers.read_frame_table(table)[0]["frame_id"] == 7


def test_zarr_sequence(tmp_path: Path):
    frames = [np.full((4, 5), i, dtype=np.uint8) for i in range(4)]
    metas = [{"frame_id": i, "pixel_format": "Mono8"} for i in range(4)]
    out = savers.save_zarr_sequence(tmp_path / "seq", frames, metas,
                                    pixel_format="Mono8", camera_settings=SETTINGS,
                                    display_settings=DISPLAY,
                                    blosc_cname="lz4", blosc_clevel=1,
                                    blosc_shuffle="shuffle")
    import zarr
    from zarr.storage import LocalStore
    g = zarr.open_group(store=LocalStore(root=out), mode="r")
    dset = g["image"]
    assert dset.shape == (4, 4, 5)
    assert np.array_equal(dset[2], np.full((4, 5), 2, dtype=np.uint8))
    meta = savers.zarr_store_meta(out)
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [1, 4, 5]
    side = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    assert side["file"]["frames"] == 4
    assert side["file"]["compressor"]["cname"] == "lz4"


# --- Frame metadata table (Parquet) -----------------------------------------

def test_sidecar_holds_globals_only_and_frames_go_to_parquet(tmp_path: Path):
    frames = [np.zeros((4, 5), dtype=np.uint16) for _ in range(4)]
    metas = [{"frame_id": i, "width": 5, "height": 4, "host_timestamp_s": 10.5 + i,
              "chunk": {"exposure_time_us": 1000.0 + i, "gain_db": 0.0}}
             for i in range(4)]
    out = savers.save_hdf5_sequence(tmp_path / "seq.h5", frames, metas,
                                    pixel_format="Mono16", camera_settings=SETTINGS)
    sidecar_path = out.with_suffix(out.suffix + ".json")
    text = sidecar_path.read_text(encoding="utf-8")
    sidecar = json.loads(text)

    # global attributes only: no per-frame rows left in the JSON
    assert sidecar["schema"] == 2
    assert "host_timestamp_s" not in text and "exposure_time_us" not in text
    assert sidecar["frame_metadata"] == {
        "format": "parquet", "file": "seq.h5.frames.parquet", "rows": 4}

    from flir_cam_gui.frame_reader import describe_source
    info = describe_source(out)
    assert len(info.frame_meta) == 4
    assert info.frame_meta[2]["frame_id"] == 2
    assert info.frame_meta[2]["host_timestamp_s"] == 12.5

    # global-for-every-frame keys are not duplicated in the table
    import pyarrow.parquet as pq
    columns = pq.read_schema(savers.frame_table_path(sidecar_path)).names
    for key in ("camera_model", "sensor", "bit_depth", "width", "height",
                "pixel_format", "acquired_shape", "acquired_dtype", "chunk"):
        assert key not in columns, key
    assert columns[0] == "frame_index" and "frame_id" in columns


def test_realistic_frame_meta_drops_every_global_column(tmp_path: Path):
    """A camera-shaped meta dict keeps only genuinely per-frame columns."""
    frames = [np.zeros((4, 5), dtype=np.uint16) for _ in range(2)]
    metas = [{"camera_model": "BFS-U3-200S6M", "sensor": "Sony IMX183",
              "bit_depth": 12, "width": 5, "height": 4,
              "pixel_format": "Mono12", "acquired_shape": [4, 5],
              "acquired_dtype": "uint16", "frame_id": i,
              "device_timestamp_ns": 1000 + i, "host_timestamp_s": 5.0 + i,
              "saturated_fraction": 0.0,
              "chunk": {"exposure_time_us": 1000.0, "gain_db": 0.0,
                        "timestamp": 1000 + i, "frame_id": i}}
             for i in range(2)]
    out = savers.save_hdf5_sequence(tmp_path / "real.h5", frames, metas,
                                    pixel_format="Mono12", camera_settings=SETTINGS)
    rows = savers.read_frame_table(savers.frame_table_path(
        out.with_suffix(out.suffix + ".json")))
    assert set(rows[0]) == {"frame_index", "frame_id", "device_timestamp_ns",
                            "host_timestamp_s", "saturated_fraction"}
    assert [r["frame_id"] for r in rows] == [0, 1]


def test_frame_table_is_small_next_to_equivalent_json(tmp_path: Path):
    """A few thousand frames must not inflate the sidecar."""
    n = 2000
    frames = [np.zeros((2, 2), dtype=np.uint8) for _ in range(n)]
    metas = [{"frame_id": i, "host_timestamp_s": 1_700_000_000.123456,
              "received_at_ns": 1_700_000_000_123_456_789 + i}
             for i in range(n)]
    out = savers.save_hdf5_sequence(tmp_path / "big.h5", frames, metas,
                                    pixel_format="Mono8", camera_settings=SETTINGS)
    sidecar_path = out.with_suffix(out.suffix + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar_path.stat().st_size < 5000          # ~2 KB, was ~250 KB
    assert sidecar["frame_metadata"]["rows"] == n
    rows = savers.read_frame_table(savers.frame_table_path(sidecar_path))
    assert len(rows) == n and rows[-1]["frame_id"] == n - 1


@pytest.mark.parametrize("kw, err", [
    ({"blosc_cname": "gzip"}, "cname"),
    ({"blosc_clevel": 5}, "clevel"),
    ({"blosc_shuffle": "lzf"}, "shuffle"),
])
def test_zarr_rejects_bad_blosc_params(tmp_path, kw, err):
    arr = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=err):
        savers.save_zarr(tmp_path / "bad", arr, pixel_format="Mono8",
                         frame_meta={}, camera_settings=SETTINGS, **kw)
