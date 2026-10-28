"""Streamed (chunked, off-thread) recording: all formats round-trip through
the Saved-images reader, early stops truncate, and bad input reports failure."""

import numpy as np
import pytest

from flir_cam_gui import stream_recorder as sr
from flir_cam_gui.frame_reader import SequenceLoader, describe_source

SETTINGS = {"camera": {"__type__": "category", "features": {
    "ExposureTime": {"kind": "float", "value": 999.0}}}}
BLOSC = {"blosc_cname": "zstd", "blosc_clevel": 2, "blosc_shuffle": "shuffle"}
FORMATS = [("hdf5_sequence", ".h5"), ("zarr_sequence", ".zarr"),
           ("npy_sequence", ".npy"), ("tiff_sequence", ".tiff")]


def _frames(n, h=4, w=5):
    return [np.full((h, w), i, dtype=np.uint16) for i in range(n)]


def _metas(n):
    return [{"frame_id": i, "pixel_format": "Mono16"} for i in range(n)]


def _stream(path, fmt, n_requested, frames, chunk=4):
    rec = sr.StreamRecorder(path, fmt, n_requested, blosc=BLOSC)
    for i in range(0, len(frames), chunk):
        rec.add(frames[i:i + chunk])
    return rec.finish(pixel_format="Mono16", camera_settings=SETTINGS,
                      display_settings={"palette": "grayscale"},
                      metas=_metas(len(frames)))


@pytest.mark.parametrize("fmt, ext", FORMATS)
def test_full_recording_round_trips(fmt, ext, tmp_path):
    frames = _frames(16)
    result = _stream(tmp_path / f"seq{ext}", fmt, 16, frames)
    assert result["ok"], result
    assert result["frames"] == 16
    info = describe_source(result["path"])
    assert info.n_frames == 16 and info.kind.endswith("sequence")
    loader = SequenceLoader(info)
    for i in range(16):
        arr, meta = loader.load(i)
        assert np.all(arr == i)
        assert meta["frame_id"] == i

    # global-only sidecar + a parquet table with one row per frame
    import json
    from pathlib import Path

    from flir_cam_gui import savers

    sidecar_path = Path(str(result["path"]) + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["frame_metadata"]["rows"] == 16
    rows = savers.read_frame_table(savers.frame_table_path(sidecar_path))
    assert len(rows) == 16 and rows[-1]["frame_id"] == 15


@pytest.mark.parametrize("fmt, ext", FORMATS)
def test_early_stop_truncates(fmt, ext, tmp_path):
    frames = _frames(5)
    result = _stream(tmp_path / f"early{ext}", fmt, 16, frames, chunk=2)
    assert result["ok"], result
    assert result["frames"] == 5
    info = describe_source(result["path"])
    assert info.n_frames == 5
    loader = SequenceLoader(info)
    for i in range(5):
        arr, _ = loader.load(i)
        assert np.all(arr == i)


def test_shape_change_fails_the_recording(tmp_path):
    rec = sr.StreamRecorder(tmp_path / "bad.npy", "npy_sequence", 10)
    rec.add([np.zeros((4, 5), np.uint16)])
    rec.add([np.zeros((4, 6), np.uint16)])
    result = rec.finish(pixel_format="Mono16", camera_settings=SETTINGS,
                        display_settings=None, metas=[])
    assert not result["ok"]
    assert "shape" in result["message"]


def test_unsupported_format_rejected(tmp_path):
    with pytest.raises(ValueError):
        sr.StreamRecorder(tmp_path / "x.raw", "raw_sequence", 10)


def test_finish_without_frames_reports_error(tmp_path):
    rec = sr.StreamRecorder(tmp_path / "n.npy", "npy_sequence", 4)
    result = rec.finish(pixel_format="Mono16", camera_settings=SETTINGS,
                        display_settings=None, metas=[])
    assert not result["ok"]
    assert not rec.is_alive()


def test_finish_waits_until_all_queued_chunks_are_written(tmp_path):
    frames = _frames(9)
    rec = sr.StreamRecorder(tmp_path / "drained.npy", "npy_sequence", 9,
                            blosc=BLOSC)
    rec.add(frames[:3])
    rec.add(frames[3:6])
    rec.add(frames[6:])

    # finish is a synchronization barrier: no chunks remain in flight and
    # the writer has closed the file before its completion result is returned.
    result = rec.finish(pixel_format="Mono16", camera_settings=SETTINGS,
                        display_settings=None, metas=_metas(9))
    assert result["ok"], result
    assert result["frames"] == 9
    assert not rec.is_alive()
    info = describe_source(result["path"])
    loader = SequenceLoader(info)
    assert info.n_frames == 9
    for i in range(9):
        arr, _ = loader.load(i)
        assert np.all(arr == i)
