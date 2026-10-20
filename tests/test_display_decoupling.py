"""Display decoupling: the live view shows the newest frame at the display
rate while acquisition (and recording) keeps running at the camera's rate.

The worker keeps a single-slot mailbox: publishing never queues more than one
GUI wake-up, and the GUI paints on its own timer instead of inside the frame
callback, so a slow repaint cannot back frames up or stall the event loop.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6", reason="PyQt6 required for display tests")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from flir_cam_gui.camera import DemoWorker  # noqa: E402
from flir_cam_gui.main import DISPLAY_INTERVAL_MS, MainWindow  # noqa: E402

FRAME = {"width": 4, "height": 4, "pixel_format": "Mono16"}


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp):
    """A clean main window, torn down inside the test.

    Leaving MainWindow instances to the garbage collector destroys a Qt widget
    tree (and its timers) at an arbitrary later moment - in this suite that
    landed in the middle of another test's paint and aborted the process.
    """
    win = MainWindow(demo_default=True)
    yield win
    win._disconnect_camera()
    win.close()
    win.deleteLater()
    qapp.processEvents()


def _frame(value: int) -> np.ndarray:
    return np.full((4, 4), value, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Worker side: single-slot mailbox
# ---------------------------------------------------------------------------

def test_publish_notifies_once_per_unconsumed_frame(qapp):
    worker = DemoWorker()
    wakes = []
    worker.frameReady.connect(lambda arr, meta: wakes.append(meta["frame_id"]))
    for fid in (1, 2, 3):
        worker._publish_frame(_frame(fid), dict(FRAME, frame_id=fid))
    assert wakes == [1]            # one wake-up, not one signal per frame
    assert worker.published_frames == 3


def test_take_latest_frame_returns_newest_and_allows_next_wake(qapp):
    worker = DemoWorker()
    wakes = []
    worker.frameReady.connect(lambda arr, meta: wakes.append(meta["frame_id"]))
    worker._publish_frame(_frame(1), dict(FRAME, frame_id=1))
    worker._publish_frame(_frame(2), dict(FRAME, frame_id=2))

    frame = worker.take_latest_frame()
    assert frame is not None and frame[1]["frame_id"] == 2
    assert worker.take_latest_frame() is None     # already caught up

    worker._publish_frame(_frame(3), dict(FRAME, frame_id=3))
    assert wakes == [1, 3]


def test_recording_keeps_every_frame_while_display_coalesces(qapp, tmp_path):
    worker = DemoWorker()
    results = []
    worker.saveDone.connect(results.append)
    worker._recording = worker._make_recording(
        str(tmp_path / "seq.npy"), 6, "npy_sequence", None)

    wakes = []
    worker.frameReady.connect(lambda arr, meta: wakes.append(meta["frame_id"]))
    for fid in range(6):
        arr = _frame(fid)
        meta = dict(FRAME, frame_id=fid)
        worker._publish_frame(arr, meta)
        worker._record_frame(arr, meta)

    assert results and results[0]["ok"], results
    assert len(wakes) == 1                      # display coalesced ...
    from flir_cam_gui.frame_reader import describe_source
    assert describe_source(results[0]["path"]).n_frames == 6   # ... recording not


def test_worker_counts_frames_the_host_never_received(qapp):
    worker = DemoWorker()
    for fid in (1, 2, 5, 6):                   # two device frames skipped
        worker._publish_frame(_frame(fid), dict(FRAME, frame_id=fid))
    assert worker.dropped_frames == 2


# ---------------------------------------------------------------------------
# GUI side: painting is deferred to the display timer
# ---------------------------------------------------------------------------

def test_display_interval_is_throttled_below_60hz(qapp):
    assert 1 <= DISPLAY_INTERVAL_MS <= 50        # at most ~60 repaints/s


def test_window_paints_newest_frame_on_the_display_timer(window):
    win = window
    win.worker = DemoWorker()                    # attached -> painting deferred
    painted = []
    win.live.show_frame = lambda arr, meta: painted.append(meta["frame_id"])

    for fid in range(5):
        win._on_frame(_frame(fid), dict(FRAME, frame_id=fid))
    assert painted == []                         # nothing painted from the slot

    win._paint_pending()
    assert painted == [4]                        # newest only
    win._paint_pending()
    assert painted == [4]                        # nothing new to paint


def test_window_ignores_a_stale_frame_when_a_newer_one_is_waiting(window):
    win = window
    worker = DemoWorker()
    win.worker = worker
    for fid in (7, 8, 9):
        worker._publish_frame(_frame(fid), dict(FRAME, frame_id=fid))

    painted = []
    win.live.show_frame = lambda arr, meta: painted.append(meta["frame_id"])
    win._on_frame(_frame(7), dict(FRAME, frame_id=7))   # stale queued frame
    win._paint_pending()
    assert painted == [9]


def test_fps_label_uses_the_acquisition_rate(window):
    win = window
    win._on_frame(_frame(1), dict(FRAME, frame_id=1, acquisition_fps=118.5))
    win._update_fps()
    assert win.lbl_fps.text() == "118.5"


def test_disconnect_clears_the_pending_frame(window):
    win = window
    win.worker = DemoWorker()
    win._on_frame(_frame(1), dict(FRAME, frame_id=1))
    assert win._pending_frame is not None
    win._disconnect_camera()
    assert win._pending_frame is None
    assert win._acq_fps is None
