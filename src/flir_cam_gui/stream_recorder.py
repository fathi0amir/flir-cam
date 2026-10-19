"""Chunked, off-thread recording.

A :class:`StreamRecorder` owns one background thread.  The acquisition worker
hands it frames in chunks (``add``) and never waits for disk; when the
recording stops, ``finish`` feeds the remaining tail through, finalizes the
file and returns the same ``saveDone`` result dict the buffered writers in
``savers`` produce.

The streamed files must stay readable by ``frame_reader``, so their layout
matches the buffered writers exactly; instead of duplicating those format
details, they are reused from ``savers`` (a deliberate, documented coupling).
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import savers


def _sidecar(path: Path, kind: str, frame_shape: tuple, dtype, n_frames: int,
             pixel_format: str, camera_settings: Dict[str, Any],
             display_settings: Optional[Dict[str, Any]],
             metas: Sequence[Dict[str, Any]]) -> None:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    channels = int(frame_shape[2]) if len(frame_shape) == 3 else 1
    savers.write_sidecar(
        path.with_suffix(path.suffix + ".json"),
        savers.build_sidecar(
            image_info={"name": path.name, "type": kind, "width": w,
                        "height": h, "channels": channels,
                        "dtype": str(dtype), "frames": n_frames,
                        "pixel_format": pixel_format},
            frame_meta={"frames": [dict(m or {}) for m in metas]},
            camera_settings=camera_settings,
            display_settings=display_settings))


class _Appender:
    """Incremental writer for one format; opened on the first frame."""

    def __init__(self, path: Path, n_frames: int, blosc: Dict[str, Any]):
        self.path = Path(path)
        self.n_frames = int(n_frames)
        self.blosc = blosc
        self._open = None            # replaced by the subclass opener
        self.count = 0

    # -- subclass interface -------------------------------------------------
    def _open_file(self, first: np.ndarray, shape: tuple) -> None: ...
    def _write_chunk(self, stack: np.ndarray, start: int) -> None: ...
    def _close(self, pixel_format, camera_settings, display_settings,
               metas) -> Path:
        return self.path

    def add(self, frames: List[np.ndarray]) -> None:
        frames = [np.ascontiguousarray(a) for a in frames]
        shapes = {a.shape for a in frames}
        if len(shapes) != 1:
            raise ValueError("mixed frame shapes inside a chunk: "
                             f"{[tuple(a.shape) for a in frames]}")
        if self._open is None:
            self._shape = frames[0].shape
            self._open_file(frames[0], self._shape)
        if frames[0].shape != self._shape:
            raise ValueError("frame shape changed mid-recording: "
                             f"{self._shape} -> {frames[0].shape}")
        self._write_chunk(np.stack(frames), self.count)
        self.count += len(frames)

    def finish(self, pixel_format: str, camera_settings: Dict[str, Any],
               display_settings: Optional[Dict[str, Any]],
               metas: Sequence[Dict[str, Any]]) -> Path:
        out = self._close(pixel_format, camera_settings, display_settings,
                          metas)
        _sidecar(out, self.SIDECAR_KIND, self._shape or (0, 0), self.DTYPE,
                 self.count, pixel_format, camera_settings,
                 display_settings, metas)
        return out


class _Hdf5Appender(_Appender):
    SIDECAR_KIND = "hdf5_sequence"

    def _open_file(self, first, shape):
        import h5py
        self.DTYPE = first.dtype
        self._f = h5py.File(self.path, "w")
        self._dset = self._f.create_dataset(
            "images", shape=(0, *shape), maxshape=(None, *shape),
            chunks=(1, *shape), compression="lzf", dtype=first.dtype)
        self._grp = self._f.require_group("metadata")
        self._open = True

    def _write_chunk(self, stack, start):
        self._dset.resize((start + stack.shape[0], *self._shape))
        self._dset[start:start + stack.shape[0]] = stack

    def _close(self, pixel_format, camera_settings, display_settings, metas):
        import json
        self._grp.attrs["pixel_format"] = pixel_format
        self._grp.attrs["frame_count"] = self.count
        self._grp.attrs["camera_settings_json"] = json.dumps(
            camera_settings, default=savers._json_default)
        if display_settings:
            self._grp.attrs["display_settings_json"] = json.dumps(
                display_settings, default=savers._json_default)
        self._f.attrs["application"] = savers.APP_NAME
        self._f.attrs["saved_at_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for key, value in savers.flatten_settings(camera_settings).items():
            self._f.attrs[f"cam/{key}"] = value
        self._f.close()
        self._open = None
        return self.path


class _ZarrAppender(_Appender):
    SIDECAR_KIND = "zarr_sequence"

    def _open_file(self, first, shape):
        import zarr
        from zarr.storage import LocalStore
        self.DTYPE = first.dtype
        params = {k.replace("blosc_", ""): v for k, v in self.blosc.items()}
        params.setdefault("cname", "lz4")
        params.setdefault("clevel", 1)
        params.setdefault("shuffle", "shuffle")
        self.group = zarr.open_group(store=LocalStore(root=self.path), mode="w")
        self.arr = self.group.create_array(
            "image", shape=(0, *shape), dtype=first.dtype,
            chunks=(1, *shape), compressors=[savers.blosc_codec(**params)],
            overwrite=True)
        self._open = True

    def _write_chunk(self, stack, start):
        self.arr.resize((start + stack.shape[0], *self._shape))
        self.arr[start:start + stack.shape[0]] = stack

    def _close(self, pixel_format, camera_settings, display_settings, metas):
        savers._zarr_attrs(self.group, pixel_format=pixel_format,
                           camera_settings=camera_settings,
                           display_settings=display_settings,
                           frame_meta={"frames": [dict(m or {}) for m in metas]},
                           array=self.arr)
        self._open = None
        return self.path


class _NpyAppender(_Appender):
    SIDECAR_KIND = "npy_sequence"

    def _open_file(self, first, shape):
        if self.n_frames <= 0:
            raise ValueError("streamed .npy needs the frame count up front")
        self.DTYPE = first.dtype
        self.part = Path(str(self.path) + ".part")
        self.mm = np.lib.format.open_memmap(
            self.part, mode="w+", dtype=first.dtype,
            shape=(self.n_frames, *shape))
        self._open = True

    def _write_chunk(self, stack, start):
        self.mm[start:start + stack.shape[0]] = stack

    def _close(self, pixel_format, camera_settings, display_settings, metas):
        self.mm.flush()
        del self.mm
        if self.count == self.n_frames:
            os.replace(self.part, self.path)
        else:
            # Early stop: materialize a real (count, H, W) .npy header once.
            mm = np.lib.format.open_memmap(self.part, mode="r")
            np.save(self.path, mm[:self.count], allow_pickle=False)
            del mm
            self.part.unlink()
        self._open = None
        return self.path


class _TiffAppender(_Appender):
    SIDECAR_KIND = "bigtiff_sequence"

    def _open_file(self, first, shape):
        import tifffile
        self.DTYPE = first.dtype
        self.tw = tifffile.TiffWriter(self.path, bigtiff=True)
        self._open = True

    def _write_chunk(self, stack, start):
        for frame in stack:
            self.tw.write(frame[None],
                          photometric="minisblack" if frame.ndim < 3 else None)

    def _close(self, pixel_format, camera_settings, display_settings, metas):
        self.tw.close()
        self._open = None
        return self.path


_APPENDERS = {"hdf5_sequence": _Hdf5Appender, "zarr_sequence": _ZarrAppender,
              "npy_sequence": _NpyAppender, "tiff_sequence": _TiffAppender}


class StreamRecorder(threading.Thread):
    """Background chunk writer; acquisition never waits for disk.

    The acquirer calls :meth:`add` (one chunk of frames) and finally
    :meth:`finish`.  Results are exactly like the buffered writers:
    ``{"ok": bool, "path": str, "message": str}``.
    """

    def __init__(self, path: Path, fmt: str, n_frames: int,
                 blosc: Optional[Dict[str, Any]] = None):
        super().__init__(daemon=True, name=f"streamWriter-{Path(path).name}")
        if fmt not in _APPENDERS:
            raise ValueError(f"streaming is not supported for {fmt}")
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._results: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._appender_cls = _APPENDERS[fmt]
        self._path = Path(path)
        self._fmt = fmt
        self._n_frames = int(n_frames)
        self._blosc = dict(blosc or {})
        self._error: Optional[BaseException] = None
        self._extra: Dict[str, Any] = {}   # finished-file details for result
        self.start()

    # -- producer side (acquisition thread) ---------------------------------
    def add(self, frames: Sequence[np.ndarray]) -> None:
        self._q.put(("add", [np.ascontiguousarray(a) for a in frames]))

    def finish(self, *, pixel_format: str, camera_settings: Dict[str, Any],
               display_settings: Optional[Dict[str, Any]],
               metas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        # Queue ordering guarantees every previously submitted chunk is
        # consumed before the finish marker.  Join before reporting completion
        # so acquisition shutdown cannot race the final disk writes/close.
        self._q.put(("finish", pixel_format, camera_settings,
                     display_settings, list(metas)))
        self.join()
        return self.result()

    def result(self, timeout: float = 120.0) -> Dict[str, Any]:
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return {"ok": False, "path": str(self._path),
                    "message": "writer thread did not report a result"}

    # -- consumer side (writer thread) ---------------------------------------
    def run(self) -> None:
        appender: Optional[_Appender] = None
        try:
            while True:
                item = self._q.get()
                if item[0] == "finish":
                    _, pixel_format, settings, display, metas = item
                    if self._error is not None:
                        raise self._error
                    if appender is None:
                        raise RuntimeError("recording stopped with no frames")
                    self._extra["attrs"] = (pixel_format, settings, display,
                                            metas)
                    out = appender.finish(pixel_format, settings, display,
                                          metas)
                    self._results.put({"ok": True, "path": str(out),
                                       "message": f"Recorded: {out}",
                                       "frames": appender.count})
                    return
                if appender is None:
                    appender = self._appender_cls(self._path, self._n_frames,
                                                  self._blosc)
                try:
                    appender.add(item[1])
                except Exception as exc:
                    self._error = exc          # keep draining; fail at finish
        except Exception as exc:
            if appender is None:
                self._error = self._error or exc
            else:
                self._error = self._error or exc
        finally:
            if appender is not None and getattr(appender, "_open", None):
                try:
                    self._cleanup(appender)
                except Exception:
                    pass
            if self._error is not None:
                self._results.put({"ok": False, "path": str(self._path),
                                   "message": f"Record failed: {self._error}"})

    @staticmethod
    def _cleanup(appender: _Appender) -> None:
        if isinstance(appender, _Hdf5Appender):
            appender._f.close()
        elif isinstance(appender, _NpyAppender) and getattr(appender, "mm", None):
            appender.mm.flush()
            del appender.mm
        elif isinstance(appender, _TiffAppender):
            appender.tw.close()
