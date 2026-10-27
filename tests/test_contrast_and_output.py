"""Display contrast controls and the output path fields.

Contrast is display-only: it never changes exposure/gain, and the live view
can be either auto-levelled (with a configurable white point) or pinned to the
limits the user sets.  Saving and recording take their target from the UI
fields, so no file dialog opens for each frame.
"""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6", reason="PyQt6 required for GUI tests")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from flir_cam_gui.live_view import LiveViewWidget  # noqa: E402
from flir_cam_gui.main import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _StubWorker:
    """Collects submitted commands; enough of the worker API for teardown."""

    def __init__(self):
        self.calls = []

    def submit(self, cmd):
        self.calls.append(cmd)

    def set_display_settings(self, settings):
        self.display_settings = dict(settings)

    def stop(self):
        pass

    def wait(self, timeout=None):
        return True


@pytest.fixture()
def window(qapp):
    win = MainWindow(demo_default=True)
    yield win
    win._disconnect_camera()
    win.close()
    win.deleteLater()
    qapp.processEvents()


META = {"frame_id": 1, "width": 64, "height": 64, "pixel_format": "Mono16"}


def _gradient(rows=64, cols=64):
    """0..4000 ramp, computed in int64 so uint16 cannot wrap."""
    values = np.arange(rows * cols, dtype=np.int64) * 4000 // (rows * cols - 1)
    return values.astype(np.uint16).reshape(rows, cols)


# ---------------------------------------------------------------------------
# LiveViewWidget: manual and auto contrast
# ---------------------------------------------------------------------------

def test_manual_levels_are_exact_and_persist(qapp):
    view = LiveViewWidget()
    arr = _gradient()
    view.show_frame(arr, META)

    view.set_bit_depth(12, auto_levels=False, levels=(300, 2500),
                       auto_max_pct=100.0)   # auto max is ignored in manual
    assert [float(v) for v in view.img.levels] == [300.0, 2500.0]

    view.show_frame(arr // 2, META)          # a new frame must not re-level
    assert [float(v) for v in view.img.levels] == [300.0, 2500.0]

    # Out-of-range limits are clamped to the bit depth, never empty/inverted.
    view.set_bit_depth(8, auto_levels=False, levels=(5000, 300))
    lo, hi = (float(v) for v in view.img.levels)
    assert 0 <= lo < hi <= 255


def test_auto_contrast_white_point_is_configurable(qapp):
    view = LiveViewWidget()
    arr = _gradient(rows=100, cols=100)
    view.show_frame(arr, {"frame_id": 1, "width": 100, "height": 100,
                          "pixel_format": "Mono16"})

    view.set_bit_depth(12, auto_levels=True, auto_max_pct=90.0)
    lo, hi = (float(v) for v in view.img.levels)
    assert hi == pytest.approx(float(np.percentile(arr, 90.0)), rel=1e-6)
    assert hi < 4000

    view.set_bit_depth(12, auto_levels=True, auto_max_pct=100.0)
    assert float(view.img.levels[1]) == pytest.approx(4000.0)
    view.set_bit_depth(12, auto_levels=True, auto_max_pct=99.5)
    assert float(view.img.levels[1]) < 4000


def test_switching_back_to_auto_recomputes_immediately(qapp):
    view = LiveViewWidget()
    arr = _gradient()
    view.show_frame(arr, META)
    view.set_bit_depth(12, auto_levels=False, levels=(10, 20))
    assert [float(v) for v in view.img.levels] == [10.0, 20.0]
    view.set_bit_depth(12, auto_levels=True)
    lo, hi = (float(v) for v in view.img.levels)
    assert hi > 3500 and lo < 100       # back on the frame's own histogram


# ---------------------------------------------------------------------------
# Main window contrast controls
# ---------------------------------------------------------------------------

def test_contrast_toggle_and_sliders_control_the_live_view(window):
    win = window
    arr = _gradient(rows=48, cols=64)
    win._on_frame(arr, {"frame_id": 1, "width": 64, "height": 48,
                        "pixel_format": "Mono16"})

    assert win.contrast_auto.isChecked()
    assert win.auto_max.isEnabled()
    assert not win.levels_min.isEnabled() and not win.levels_max.isEnabled()

    win.auto_max.setValue(90.0)
    hi = float(win.live.img.levels[1])
    assert hi == pytest.approx(float(np.percentile(arr, 90.0)), rel=1e-6)

    win.contrast_auto.setChecked(False)
    assert not win.auto_max.isEnabled()
    assert win.levels_min.isEnabled() and win.levels_max.isEnabled()
    win.levels_min_value.setValue(400)
    win.levels_max_value.setValue(3000)
    assert [float(v) for v in win.live.img.levels] == [400.0, 3000.0]

    # Dragging max below min keeps the pair ordered and the levels exact.
    win.levels_max_value.setValue(200)
    lo, hi = (float(v) for v in win.live.img.levels)
    assert lo < hi
    assert [lo, hi] == [float(win.levels_min_value.value()),
                        float(win.levels_max_value.value())]

    settings = win.current_display_settings()
    assert settings["auto_levels"] is False
    assert settings["levels"] == [int(lo), int(hi)]
    assert settings["auto_max_pct"] == pytest.approx(90.0)


def test_contrast_mode_is_remembered_by_the_saved_images_view(window):
    win = window
    win.contrast_auto.setChecked(False)
    win.levels_min_value.setValue(100)
    win.levels_max_value.setValue(2000)
    stored = win.reader_tab._display_settings
    assert stored["auto_levels"] is False
    assert stored["bit_depth"] == 12
    assert stored["levels"] == [100, 2000]

    win.bits_combo.setCurrentIndex(win.bits_combo.findData(8))
    assert win.levels_max_value.maximum() == 255
    assert win.levels_max_value.value() <= 255


# ---------------------------------------------------------------------------
# Output path fields
# ---------------------------------------------------------------------------

def test_save_uses_the_path_fields_without_a_dialog(window, tmp_path):
    win = window
    win.worker = _StubWorker()
    win.out_dir_edit.setText(str(tmp_path / "run"))
    win.out_name_edit.setText("frame")

    # The button lives in the pinned action row, below the scrolling settings;
    # clicking it must still reach the worker.
    win.save_fmt_combo.setCurrentIndex(1)       # HDF5
    win.save_btn.setEnabled(True)
    win.save_btn.click()
    cmd = win.worker.calls[-1]
    assert cmd["action"] == "save_last"
    assert Path(cmd["path"]) == tmp_path / "run" / "frame.h5"
    assert (tmp_path / "run").is_dir()      # the folder is created for us

    # A typed extension is replaced by the format's own suffix.
    win.out_name_edit.setText("stack.h5")
    win._save_last(fmt="raw")
    assert Path(win.worker.calls[-1]["path"]) == tmp_path / "run" / "stack.raw"

    # An existing target is never overwritten silently.
    (tmp_path / "run" / "stack.raw").write_bytes(b"old")
    win._save_last(fmt="raw")
    assert Path(win.worker.calls[-1]["path"]) == tmp_path / "run" / "stack_001.raw"


def test_record_uses_the_path_fields_and_a_unique_folder(window, tmp_path):
    win = window
    win.worker = _StubWorker()
    win.out_dir_edit.setText(str(tmp_path))
    win.out_name_edit.setText("burst")
    win.rec_fmt_combo.setCurrentIndex(1)        # raw sequence -> one folder
    win.rec_btn.setEnabled(True)
    win.rec_btn.click()
    cmd = win.worker.calls[-1]
    assert cmd["action"] == "record"
    assert Path(cmd["path"]) == tmp_path / "burst_seq"

    # A completed raw recording leaves that folder behind: the next one must
    # pick a fresh name instead of mixing frames into it.
    (tmp_path / "burst_seq").mkdir()
    win._recording = False
    win.rec_btn.click()
    assert Path(win.worker.calls[-1]["path"]) == tmp_path / "burst_001_seq"


def test_pixel_depth_enum_clamps_the_sliders_to_the_bitness(window):
    """10-bit data in a uint16 container must use 0-1023, not 0-65535."""
    win = window
    win._on_frame(_gradient(), {"frame_id": 1, "width": 64, "height": 48,
                                "pixel_format": "Mono16"})
    win.contrast_auto.setChecked(False)

    win.bits_combo.setCurrentIndex(win.bits_combo.findData(16))
    assert win.levels_min.maximum() == 65535
    assert win.levels_max.maximum() == 65535
    win.levels_min_value.setValue(1000)
    win.levels_max_value.setValue(60000)

    win.bits_combo.setCurrentIndex(win.bits_combo.findData(10))
    assert win.levels_min.maximum() == 1023
    assert win.levels_max.maximum() == 1023
    assert win.levels_max_value.maximum() == 1023
    assert (win.levels_min_value.value(), win.levels_max_value.value()) == (1000, 1023)
    assert [float(v) for v in win.live.img.levels] == [1000.0, 1023.0]
    assert win.current_display_settings()["bit_depth"] == 10

    # The labels say which range is in force.
    assert "0-1023" in win.bits_combo.currentText()
    assert "0-1023" in win.levels_max_value.toolTip()


def test_full_range_follows_the_selected_pixel_depth(window):
    win = window
    win.contrast_auto.setChecked(False)
    assert (win.levels_min_value.value(), win.levels_max_value.value()) == (0, 4095)

    win.bits_combo.setCurrentIndex(win.bits_combo.findData(8))
    assert (win.levels_min_value.value(), win.levels_max_value.value()) == (0, 255)
    assert win.levels_max.value() == 255

    win.bits_combo.setCurrentIndex(win.bits_combo.findData(16))
    assert (win.levels_min_value.value(), win.levels_max_value.value()) == (0, 65535)
    assert win.levels_max.value() == 65535

    # A custom window is kept when it still fits, and clamped when it does not.
    win.levels_min_value.setValue(100)
    win.levels_max_value.setValue(50000)
    win.bits_combo.setCurrentIndex(win.bits_combo.findData(12))
    assert (win.levels_min_value.value(), win.levels_max_value.value()) == (100, 4095)


def test_record_suffix_follows_the_selected_format(window, tmp_path):
    win = window
    win.worker = _StubWorker()
    win.out_dir_edit.setText(str(tmp_path))
    win.out_name_edit.setText("cap")
    win.rec_fmt_combo.setCurrentIndex(0)        # HDF5 sequence
    win._toggle_record()
    assert Path(win.worker.calls[-1]["path"]) == tmp_path / "cap.h5"


def test_browse_button_uses_the_native_folder_dialog(window, tmp_path, monkeypatch):
    from flir_cam_gui import main as main_mod

    seen = {}

    class _Dialog:
        @staticmethod
        def getExistingDirectory(parent, title, start=""):
            seen.update(parent=parent, title=title, start=start)
            return str(tmp_path / "picked")

    monkeypatch.setattr(main_mod, "QFileDialog", _Dialog)
    window.browse_dir_btn.click()
    assert window.out_dir_edit.text() == str(tmp_path / "picked")
    assert seen["parent"] is window
    assert seen["title"]


def test_output_hint_previews_both_targets(window, tmp_path):
    win = window
    win.out_dir_edit.setText(str(tmp_path))
    win.out_name_edit.setText("cap")
    win.save_fmt_combo.setCurrentIndex(1)       # HDF5
    win.rec_fmt_combo.setCurrentIndex(2)        # Zarr
    text = win.out_hint.text()
    assert "cap.h5" in text
    assert "cap.zarr" in text
    assert str(tmp_path) not in text        # names only, keeps the panel narrow
