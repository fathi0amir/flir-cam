# FLIR Spinnaker Camera GUI (PyQt6)

A comprehensive Python GUI for **FLIR Spinnaker cameras** (PySpin), built with
**Qt6 (PyQt6)**:

- **Multiple cameras**: every camera Spinnaker detects is listed in the
  *Camera* picker (model, serial, transport) with a **Refresh** button, so you
  pick which one to connect to; `--serial SN` preselects it from the CLI. The
  scan runs off the UI thread and the picker locks while connected.
- **Live view** at full frame rate with **crosshair line profiles** attached to
  the **top** and **right** edges of the image (profiles update as you move the
  crosshair or hover the mouse). The crosshair stays **draggable while
  acquisition is running** — incoming frames never move it — and the profile
  plots are pixel-aligned with the image.
- **Display orientation**: `as acquired` (the camera's own orientation),
  `rotate 90° CW/CCW`, `rotate 180°`, `flip horizontal/vertical`. Use it for a
  camera that is physically mounted rotated; the frame geometry readout always
  reports the orientation actually on screen. **Reset View** fits the image
  instantly (it sets the view range instead of auto-ranging over every pixel).
- **Saved images tab**: open a recorded stack and browse it — `.raw` frame or
  folder of raw frames, **HDF5** (`/image` or `/images`) or **Zarr** (`/image`)
  — with **play/pause**, playback speed, loop, a **scrubber** to jump to any
  frame, first/prev/next/last buttons, and the per-frame metadata
  (frame id, exposure, gain, saturation) from the frame table. Frames are read
  **lazily**: opening a stack reads no pixels, and only the frame on screen is
  read, with rapid scrubbing collapsed into a single read of the final
  position.
- **Full GenICam feature tree browser** — view and *modify every available
  camera node* (integer/float/boolean/enumeration/string editors, command
  execution buttons), including the transport-layer (stream) node map.
  Values refresh periodically live (temperature, counters, etc.).
- **Save the last frame several ways**:
  - `raw` binary plus a **JSON sidecar** with *every setting for that image*
    (full node-map snapshot, display settings) and a Parquet table with the
    per-frame rows.
  - **HDF5** (`/image` dataset + `metadata` group attrs + flattened camera
    settings attrs `cam/...`) plus the same JSON sidecar.
  - **Zarr v3** store (`<name>.zarr`, `/image` array) compressed with
    **Blosc** — choose **lz4** or **zstd**, compression **level 1–3**,
    **shuffle** enabled — plus the same JSON sidecar.
- **Record sequences**: N frames into one **HDF5** file (`/images`, LZF
  compressed, chunked), one **Zarr** store (frames stacked on axis 0, one
  Blosc chunk per frame), or a folder of raw frames + `sequence.json` sidecar.
  Every sequence writes its per-frame metadata to `<name>.frames.parquet`
  (columnar, one row per frame) next to the JSON sidecar. Frames are handed to
  a writer thread as they arrive, so saving never stalls acquisition, and the
  **Recording** group shows a progress bar with the frames still to capture.
  Stopping acquisition or disconnecting flushes all accepted frames to disk.
- **Menu → File → Save settings snapshot** for the raw node map dump.

The live camera view and the saved-image viewer are separate tabs, so browsing
recorded data never disturbs a running acquisition. The live view is also
decoupled from the acquisition loop: frames go through a single-slot mailbox
and are repainted at ~30 Hz, always showing the newest frame. Acquisition and
recording keep running at the camera's own rate, so a slow repaint (large
sensor, heavy GUI load) can neither drop frames nor stall the UI.

If the **FPS** readout grows a `N dropped` suffix, the camera is producing
frames faster than this host can receive them. Lower `AcquisitionFrameRate`
(enable `AcquisitionFrameRateEnable` first) to a value at or below the rate
the host sustains — around 85 fps on the test machine — for a drop-free
stream; the tooltip on the FPS label spells this out.

## Install (uv)

The environment is **pinned to Python 3.12** (`.python-version`, and the wheel
is cp312) and fully locked by `uv.lock`.

Before running any `uv` commands, install the **FLIR Spinnaker SDK 4.4** for
your system and place its matching PySpin wheel in `vendor/` as
`spinnaker_python-4.4.0.246-cp312-cp312-win_amd64.whl`. This repository does
not include the wheel. The filename is configured as the local package source
in `pyproject.toml`; it must match your Python 3.12 and Windows x64 environment.

```bash
uv sync                          # GUI deps (PyQt6, pyqtgraph, numpy, h5py, zarr)
uv sync --extra spinnaker        # + vendored FLIR PySpin 4.4.0.246 wheel
```

PySpin is not on PyPI; it is installed from the locally supplied wheel. See the
[FLIR Spinnaker SDK 4.4](https://www.flir.com/products/spinnaker-sdk/)
for the Windows installer and SDK distribution.

Verify: `uv run python -c "import PySpin; print('pyspin ok')"`

## Run

```bash
uv run flir-cam            # real camera if PySpin is working; simulation otherwise
uv run flir-cam --live     # force hardware mode (errors if no PySpin)
uv run flir-cam --demo     # force simulation mode
uv run flir-cam --open seq.zarr   # open a saved stack in the Saved images tab
uv run flir-cam --serial 12345678 # preselect that camera in the picker
```

Simulated camera modules (approximate public sensor specs, offline only):

```bash
uv run flir-cam --model "BFS-U3-200S6M (simulated CMV4000 CMOS, mono 12-bit)"
```

Available simulated models (see `src/flir_cam_gui/camera.py::DEMO_MODELS`):

- `BFS-U3-200S6M` — CMV4000 CMOS, mono 12-bit, **1536 x 2048 (portrait)**
- `BFS-U3-31S4M` — Sony IMX265 CMOS, mono 12-bit, 2048 x 1536
- `GS3-U3-23S6M` — Sony ICX672 CCD, mono 12-bit, 1920 x 1440
- `BFS-U3-51S5C` — 5.1 MP OSC CMOS, color (RGB8Packed), 2448 x 2048

Frames are generated with a sensor-shaped noise model: photon **shot noise**
(normal approximation of Poisson), **read noise**, **dark current**,
**baseline offset**, ADC gain scaled by the configured dB gain, and clipping to
the sensor's full scale. Raising exposure/gain saturates frames just like real
hardware, and saturation fraction is reported in frame metadata.

## Usage notes

1. **Pick a source and a camera** — the source combo selects Live or a
   simulated model; with *Live* the *Camera* combo lists every connected FLIR
   camera (`model  SN=…  [transport]`), so choose the one to open and press
   **Connect**. *Refresh* re-scans after plugging a camera in.
2. **▶ Start acquisition** — live view and profiles begin.
3. **Crosshair** — drag either line with the left mouse button: it follows the
   cursor the whole way, *while acquisition is running*, and both profiles
   update live. The press must be within a few pixels of a line to grab it;
   dragging anywhere else pans the view as usual, and `Reset View` refits the
   frame.
4. Adjust any node in the *Camera features* tree; read-only values refresh
   every few seconds (and instantly after you write a value).
5. **Saved images** tab — *Open file...* for a `.raw`/`.h5`/`.hdf5` (or its
   `.json` sidecar) and *Open folder...* for a Zarr store or a folder of raw
   frames. Play/pause, drag the scrubber, or step with `|< < > >|`; the frame
   metadata line shows what the per-frame Parquet table recorded for that frame.
   *Orientation*,
   *Contrast* and *Palette* in the Display panel apply to this viewer too.
6. **Display panel** — *Orientation* renders the frame as acquired, rotated
   (90° CW/CCW, 180°), flipped (horizontal/vertical) or transposed, for a
   camera that is physically mounted at an angle. Whatever the window shape and
   whatever orientation is picked, the image keeps the sensor's own aspect ratio
   (square pixels, never stretched) and is centred with the whole frame visible;
   *Reset View* refits it instantly.
7. **Contrast** (display only — it never changes exposure or gain):
   - *Auto contrast* on (default): each frame's black/white points come from
     its histogram. *Auto max* is the intensity percentile mapped to white
     (99.5 % by default, so a few hot pixels do not flatten the image; set it
     to 100 % to use the brightest pixel).
   - *Pixel depth* (8 / 10 / 12 / 16-bit) is the bitness of the pixel
     *values*, not the dtype of the container: a Mono10/Mono12 camera is
     delivered in a `uint16` array whose values only span 0-1023 or 0-4095.
     Selecting that bitness limits the contrast sliders to 0-255 / 0-1023 /
     0-4095 / 0-65535 instead of always exposing the full 16-bit range. If the
     handles covered the old full range they follow the new bitness; a custom
     window is clamped to it.
   - *Auto contrast* off: the *Min* and *Max* sliders (with exact spin boxes)
     are the black and white points, in the selected pixel-depth scale.
     The pair is kept ordered (min < max) as you drag.
   - Both modes apply to the live view and the Saved images viewer; the chosen
     values are recorded in the sidecar's `display_settings`.
8. **Save / record** with the right-hand panels or the File menu. The *Output
   location* panel holds the **Folder** (type it, or *Browse...* to use the OS
   folder picker) and the **File name**; the format's suffix is appended
   (`.raw`, `.h5`, `.zarr`, `.npy`, `.tiff`, or `_seq` for a raw sequence) and
   the line below previews both targets. Nothing is overwritten: an existing
   name gets `_001`, `_002`, ... instead. When it saves an
   image it **always writes a sidecar** (`<name>.json`) with the *global*
   attributes:
   - `camera_settings` — full node maps (camera + stream),
   - `file` — geometry, dtype, byte order, byte count,
   - `display_settings` — palette/bit-depth/levels used in the live view,
   - `frame_metadata` — a reference to the per-frame table (see below).

   Per-frame attributes (frame ID, device/host timestamps, saturation, and
   the raw file name for raw sequences) are **not** stored in the JSON: they go
   to a columnar **Parquet table beside it** — `<name>.json` →
   `<name>.frames.parquet` (`sequence.json` → `sequence.frames.parquet` for raw
   sequences). One row per frame, `frame_index` first:

   ```python
   import pyarrow.parquet as pq
   t = pq.read_table("seq.h5.frames.parquet")
   print(t.column_names)            # frame_index, frame_id, host_timestamp_s, ...
   print(t.column("host_timestamp_s").to_pylist()[:3])
   # or with DuckDB: SELECT frame_id, host_timestamp_s FROM 'seq.h5.frames.parquet'
   ```

   Values that are identical for every frame of a recording (`camera_model`,
   `sensor`, `bit_depth`, `width`, `height`, `pixel_format`, `acquired_shape`,
   `acquired_dtype`, `chunk`) are **not** repeated as columns; they live once in
   the sidecar's `file` / `camera_settings`.

   This keeps the sidecar constant-size: it holds only the globals (a few KB),
   while every recorded frame used to add another JSON row. Measured: 3000
   frames with three metadata fields each are ~293 KB of JSON, versus a ~0.5 KB
   sidecar plus ~51 KB of Parquet. Sidecars from older versions that embed the
   rows inline (`schema: 1`) are still read.

### Sidecar data schema

Every save writes two files: a JSON sidecar with the **global** attributes and a
Parquet table with one **row per frame**.

| data | file name |
|---|---|
| JSON sidecar | `<name>.json` — `frame.raw.json`, `seq.h5.json`, `seq.zarr.json`, … |
| frame table | `<name>.frames.parquet` — `frame.raw.frames.parquet`, `seq.h5.frames.parquet`, … |
| raw sequence | `sequence.json` + `sequence.frames.parquet` inside the `_seq` folder |

#### JSON sidecar (`schema: 2`)

Top-level keys (values abridged):

```json
{
  "schema": 2,
  "application": "flir-cam-gui",
  "app_version": "0.1.0",
  "host": "ILLab-DAFi",
  "platform": "Windows-11-10.0.26300-SP0",
  "python": "3.12.9",
  "saved_at_utc": "2026-10-28T09:23:09Z",
  "file": {
    "name": "seq.h5", "type": "hdf5_sequence",
    "width": 1536, "height": 2048, "channels": 1,
    "dtype": "uint16", "frames": 3, "pixel_format": "Mono16"
  },
  "frame_metadata": {"format": "parquet", "file": "seq.h5.frames.parquet", "rows": 3},
  "camera_settings": {"camera": {"__type__": "category", "...": "..."},
                       "stream": {"__type__": "category", "...": "..."}},
  "display_settings": {"palette": "grayscale", "follow_mouse": false,
                       "auto_levels": true, "bit_depth": 12, "auto_max_pct": 99.5,
                       "levels": null, "orientation": "as acquired"}
}
```

| key | type | meaning |
|---|---|---|
| `schema` | int | sidecar layout version, currently `2` |
| `application` | string | always `flir-cam-gui` |
| `app_version` | string | version that wrote the file |
| `host` | string | machine name |
| `platform` | string | OS string |
| `python` | string | interpreter version |
| `saved_at_utc` | string | `YYYY-MM-DDTHH:MM:SSZ` |
| `file` | object | pixel payload description (see below) |
| `frame_metadata` | object | pointer to the Parquet frame table |
| `camera_settings` | object | GenICam node-map snapshot (`camera` + `stream`) |
| `display_settings` | object | live-view settings; omitted when unknown |

`frame_metadata` points at the table next to the JSON:

| key | type | meaning |
|---|---|---|
| `format` | string | always `"parquet"` |
| `file` | string | table file name (same folder as the JSON) |
| `rows` | int | row count = number of frames |

`file` always carries `name` (file-backed saves; not raw sequences),
`type`, `width`, `height`, `channels` (3 for RGB), `dtype` and `pixel_format`,
plus per-type keys:

| `file.type` | extra keys |
|---|---|
| `raw_binary` | `byte_order`, `shape`, `bytes`, `description` |
| `raw_sequence` | `frames`, `byte_order` (frames are separate `.raw` files, so no `name`/`shape`/`bytes`) |
| `hdf5_single` | — |
| `hdf5_sequence` | `frames` |
| `zarr_single` | `format` (`zarr3`), `compressor` |
| `zarr_sequence` | `frames`, `format`, `compressor` |
| `npy` | `shape`, `bytes` |
| `npy_sequence` | `shape`, `bytes`, `frames` |
| `bigtiff` | `shape`, `bytes` |
| `bigtiff_sequence` | `shape`, `bytes`, `frames` |

`compressor` is `{"name": "blosc", "cname": "lz4"|"zstd", "clevel": 1-3, "shuffle": "shuffle"}`.

`camera_settings` is the full node map as nested categories and leaves (one
leaf shown; `name`/`tooltip` elided here):

```json
{"camera": {"__type__": "category", "display": "Root", "writable": false,
             "features": {
               "ExposureTime": {"kind": "float", "value": 5000.0,
                                "min": 10.0, "max": 1000000.0, "unit": "us",
                                "readable": true, "writable": true}}},
 "stream": {"__type__": "category", "display": "Root (TLStream)",
            "writable": false, "features": {}}}
```

Categories nest recursively via `features` and carry `__type__`, `display`,
`writable` and `features`. Every leaf has `name`, `display`, `tooltip`,
`readable`, `writable` and `kind` (`integer`, `float`, `boolean`,
`enumeration`, `string`, `command` or `value`); leaves can also carry `value`,
`min`/`max`/`inc`, `unit`, and for enumerations
`entries: [{"symbolic": ..., "value": ..., "current": bool}]`.

`display_settings` keys: `palette`, `follow_mouse`, `auto_levels`, `bit_depth`,
`auto_max_pct`, `levels` (`null` in auto mode, else `[min, max]`) and
`orientation`.

#### Parquet frame table

One row per recorded frame, Snappy-compressed, `frame_index` first:

| column | Arrow type | null | meaning |
|---|---|---|---|
| `frame_index` | int64 | no | 0-based row number; for raw sequences it matches `frame_<index>.raw` |
| `file` | string | yes | raw sequences only: the frame's `.raw` file name |
| `frame_id` | int64 | yes | frame counter reported by the camera |
| `device_timestamp_ns` | int64 | yes | timestamp reported by the device, ns |
| `received_at_ns` | int64 | no | host receive time, ns since epoch |
| `host_timestamp_s` | double | no | same instant in seconds since epoch |
| `saturated_fraction` | double | yes | fraction of full-scale pixels (simulated cameras) |
| `acquisition_fps` | double | yes | measured acquisition rate when the frame arrived |

Any other per-frame key a backend adds becomes another column.

These fields are **global** and are deliberately not repeated per row (they live
in `file` / `camera_settings`): `camera_model`, `sensor`, `bit_depth`, `width`,
`height`, `pixel_format`, `acquired_shape`, `acquired_dtype`, `chunk`.

```python
import pyarrow.parquet as pq

t = pq.read_table("seq.h5.frames.parquet")
print(t.schema)
print(t.to_pandas().head())     # or: t.to_pylist()
```

Version note: `schema: 1` sidecars (older recordings) kept the per-frame rows
inline under `frame_metadata.frames`; the GUI reads both layouts.

### Reading a raw file back

The easiest way is the **Saved images** tab in the GUI. To do it in Python:

```python
import numpy as np, json, pathlib
meta = json.loads(pathlib.Path("frame.raw.json").read_text())["file"]
arr = np.fromfile("frame.raw", dtype=np.dtype(meta["dtype"])
                  ).reshape(meta["height"], meta["width"])
```

For a colour frame add the channel axis
(`meta["channels"]`, stored last: `(height, width, channels)`).

### Reading a HDF5 or Zarr store back

```python
import h5py, zarr
from zarr.storage import LocalStore

with h5py.File("seq.hdf5", "r") as f:
    stack = f["images"]            # (N, H, W): one chunk per frame
    frame = stack[3]
img = zarr.open_group(store=LocalStore(root="seq.zarr"), mode="r")["image"]  # (N, H, W)
```

## Development

```bash
uv sync              # installs dev deps (pytest) too
uv run pytest -q     # unit + offscreen GUI smoke tests
uv run flir-cam --demo   # manual GUI run
```

Architecture note: All Spinnaker/GenICam access happens inside a worker thread
(`src/flir_cam_gui/camera.py`); the UI communicates only via commands and Qt
signals, so camera nodes are never touched from the UI thread. Stack reading
(`src/flir_cam_gui/frame_reader.py`) keeps HDF5/Zarr handles open but reads a
single frame per request, so a 10 000 frame stack costs the same as a 10 frame
one.
