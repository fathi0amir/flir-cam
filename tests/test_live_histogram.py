"""Live-view histogram toggling (offscreen)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from flir_cam_gui.live_view import LiveViewWidget  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def view(qapp):
    v = LiveViewWidget()
    frame = np.arange(3000, dtype=np.uint16).reshape(50, 60)
    v.show_frame(frame)
    return v


def test_histogram_hidden_by_default(view):
    assert not view._hist_visible
    assert not view._hist_added


def test_histogram_toggle_on_and_off(view):
    view.set_histogram_visible(True)
    assert view._hist_visible and view._hist_added
    assert view._hist.isVisible()

    view.set_histogram_visible(False)
    assert not view._hist_visible
    assert not view._hist.isVisible()
    assert view._hist_added       # stays constructed, just hidden


def test_histogram_tracks_palette(view):
    view.set_histogram_visible(True)
    view.set_palette("viridis")   # must not raise with the histogram attached
    view.set_palette("grayscale")


def test_histogram_survives_new_frames(view):
    view.set_histogram_visible(True)
    view.show_frame(np.zeros((40, 50), dtype=np.uint16))
    assert view._hist.isVisible()


def test_clear_frame_keeps_histogram_state(view):
    view.set_histogram_visible(True)
    view.clear_frame()
    assert view._hist_visible    # toggle intent survives a disconnect
    view.show_frame(np.zeros((40, 50), dtype=np.uint16))
    assert view._hist.isVisible()
