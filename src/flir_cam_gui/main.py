"""FLIR Spinnaker camera GUI (PyQt6) — main window & entry point."""

from __future__ import annotations

import argparse
import faulthandler
import sys
import time
from enum import IntEnum
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QStandardPaths, QTimer, pyqtSlot
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget,
                             QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
                             QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                             QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout,
                             QWidget)

from .camera import DEMO_MODELS, PYSPIN_AVAILABLE, CameraLister, create_worker
from .live_view import ORIENTATIONS, PALETTES, LiveViewWidget
from .node_browser import NodeBrowser
from .savers import write_sidecar
from . import frame_reader

APP_TITLE = "FLIR Spinnaker Camera GUI"

#: live-view repaint period (ms).  Acquisition and recording run at the
#: camera's own rate; the GUI only ever paints the newest frame, so a slow
#: repaint never stalls the UI event loop or backs frames up in a queue.
DISPLAY_INTERVAL_MS = 33          # ~30 repaints/s

#: suffix appended to the typed file name for each save / sequence format.
#: (.raw/.h5/... typed by the user is stripped first, so "stack.h5" never
#: becomes "stack.h5.h5".)
_SAVE_SUFFIX = {"raw": ".raw", "hdf5": ".h5", "zarr": ".zarr",
                "npy": ".npy", "tiff": ".tiff"}
_RECORD_SUFFIX = {"hdf5_sequence": ".h5", "raw_sequence": "_seq",
                  "zarr_sequence": ".zarr", "npy_sequence": ".npy",
                  "tiff_sequence": ".tiff"}
_IMAGE_SUFFIXES = {".raw", ".h5", ".hdf5", ".zarr", ".npy", ".tif", ".tiff"}


class PixelBitDepth(IntEnum):
    """Significant bits of the pixel values, not the container width.

    Spinnaker delivers Mono10/Mono12 frames in a uint16 container: the values
    themselves only span 0..1023 / 0..4095.  The display contrast must work in
    that pixel bitness, not in 16-bit, or the manual limits become unusable.
    """

    EIGHT = 8
    TEN = 10
    TWELVE = 12
    SIXTEEN = 16

    @property
    def max_level(self) -> int:
        return (1 << int(self)) - 1

    @property
    def label(self) -> str:
        return f"{int(self)}-bit (0-{self.max_level})"


class MainWindow(QMainWindow):
    def __init__(self, demo_default: bool = False, model_default: Optional[str] = None,
                 serial_default: Optional[str] = None):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 950)

        self.worker = None
        self._last_tree: dict = {}
        self._frame_times: list[float] = []   # display fallback only
        self._acq_fps: Optional[float] = None  # measured on the acquisition thread
        self._pending_frame: Optional[tuple] = None
        self._frames_since_meta = 0
        self._recording = False
        self._lister: Optional[CameraLister] = None
        self._cameras: list = []          # last enumeration result
        self._wanted_serial = serial_default   # --serial hint, consumed once
        self._levels_top: Optional[int] = None  # previous contrast range

        self._build_ui()
        self._connect_ui()

        # periodic value refresh (live values like temperature / counters)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(3000)
        self._refresh_timer.timeout.connect(self._poll_feature_values)

        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(500)
        self._fps_timer.timeout.connect(self._update_fps)

        # Repaints are decoupled from acquisition: the timer drains the newest
        # frame at the display rate, so heavy frames cannot starve the UI.
        self._paint_timer = QTimer(self)
        self._paint_timer.setInterval(DISPLAY_INTERVAL_MS)
        self._paint_timer.timeout.connect(self._paint_pending)
        self._record_progress_timer = QTimer(self)
        self._record_progress_timer.setInterval(100)
        self._record_progress_timer.timeout.connect(self._refresh_record_progress)

        # Start completely disconnected - user must manually connect
        if demo_default or not PYSPIN_AVAILABLE:
            key = model_default or next(iter(DEMO_MODELS))
            self.source_combo.setCurrentText(f"Simulate: {key}")
        else:
            self.source_combo.setCurrentText("Live (Spinnaker)")
        self._sync_camera_picker()
        if not self._current_is_demo():
            self._refresh_cameras()      # populate the picker before connecting

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        # --- source bar ---
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Camera source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("Live (Spinnaker)")
        for key in DEMO_MODELS:
            self.source_combo.addItem(f"Simulate: {key}")
        src_row.addWidget(self.source_combo, 2)
        src_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Searching for cameras\u2026")
        self.camera_combo.setToolTip(
            "FLIR cameras detected by Spinnaker - pick which one to connect to")
        self.refresh_cameras_btn = QPushButton("Refresh")
        self.refresh_cameras_btn.setToolTip("Re-scan for connected cameras")
        self.refresh_cameras_btn.clicked.connect(self._refresh_cameras)
        src_row.addWidget(self.camera_combo, 2)
        src_row.addWidget(self.refresh_cameras_btn)
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        src_row.addWidget(self.connect_btn)
        src_row.addWidget(self.disconnect_btn)
        self.start_btn = QPushButton("▶ Start acquisition")
        self.start_btn.setEnabled(False)
        self.stop_btn = QPushButton("■ Stop acquisition")
        self.stop_btn.setEnabled(False)
        src_row.addWidget(self.start_btn)
        src_row.addWidget(self.stop_btn)
        src_row.addStretch(1)
        v.addLayout(src_row)

        # --- splitter: settings | centre tabs | right panel ---
        split = QSplitter()
        v.addWidget(split, 1)

        left = QTabWidget()
        self.browser = NodeBrowser()
        left.addTab(self.browser, "Camera features")
        split.addWidget(left)

        # Live acquisition and saved-stack playback each own a viewer, so
        # browsing a recorded stack never disturbs the running camera view.
        self.live = LiveViewWidget()
        self.reader_tab = frame_reader.FrameReaderTab()
        self.centre_tabs = QTabWidget()
        self.centre_tabs.addTab(self.live, "Live view")
        self.centre_tabs.addTab(self.reader_tab, "Saved images")
        split.addWidget(self.centre_tabs)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(4, 4, 4, 4)

        # frame info
        info_grp = QGroupBox("Image / camera info")
        info_form = QFormLayout(info_grp)
        self.lbl_frame = QLabel("-")
        self.lbl_dims = QLabel("-")
        self.lbl_format = QLabel("-")
        self.lbl_fps = QLabel("-")
        self.lbl_cross = QLabel("-")
        self.lbl_state = QLabel("disconnected")
        for a, w in (("Frame", self.lbl_frame), ("Size", self.lbl_dims), ("Format", self.lbl_format),
                     ("FPS", self.lbl_fps), ("Crosshair", self.lbl_cross), ("State", self.lbl_state)):
            w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            info_form.addRow(a, w)
        right_lay.addWidget(info_grp)

        # display settings
        disp_grp = QGroupBox("Display")
        disp_form = QFormLayout(disp_grp)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(PALETTES)
        self.palette_combo.setCurrentIndex(0)
        self.palette_combo.currentTextChanged.connect(self._palette_changed)
        self.bits_combo = QComboBox()
        for depth in PixelBitDepth:
            # Store a plain int: Qt's findData compares QVariants and would not
            # match an IntEnum against 8/10/12/16.
            self.bits_combo.addItem(depth.label, int(depth))
        self.bits_combo.setCurrentIndex(
            self.bits_combo.findData(int(PixelBitDepth.TWELVE)))  # 12-bit sensors
        self.bits_combo.setToolTip(
            "Pixel bitness of the frame values. A camera may deliver 10- or "
            "12-bit data in a 16-bit container; select that bitness here so "
            "the contrast limits cover 0-1023 / 0-4095, not 0-65535.")
        self.bits_combo.currentTextChanged.connect(self._bits_changed)
        # Contrast is display-only: it never touches exposure or gain.  Auto
        # mode picks the levels per frame; manual mode uses exactly the limits
        # the user sets here.
        self.contrast_auto = QCheckBox("Auto contrast")
        self.contrast_auto.setChecked(True)
        self.contrast_auto.setToolTip(
            "Off: the display maps [min, max] to black/white exactly as set "
            "below. On: the limits are computed from every frame.")
        self.auto_max = QDoubleSpinBox()
        self.auto_max.setRange(50.0, 100.0)
        self.auto_max.setDecimals(1)
        self.auto_max.setSingleStep(0.5)
        self.auto_max.setValue(99.5)
        self.auto_max.setSuffix(" %")
        self.auto_max.setKeyboardTracking(False)
        self.auto_max.setToolTip(
            "Auto contrast white point: the intensity percentile mapped to "
            "white (black is the 0.5th percentile). 99.5 % ignores a few "
            "hot pixels; 100 % maps the brightest pixel to white.")
        self.levels_min = QSlider(Qt.Orientation.Horizontal)
        self.levels_min_value = QSpinBox()
        self.levels_max = QSlider(Qt.Orientation.Horizontal)
        self.levels_max_value = QSpinBox()
        for slider, spin, top in ((self.levels_min, self.levels_min_value, 0),
                                  (self.levels_max, self.levels_max_value, 4095)):
            slider.setRange(0, 4095)
            slider.setValue(top)
            spin.setRange(0, 65535)
            spin.setValue(top)
            spin.setKeyboardTracking(False)
            slider.valueChanged.connect(spin.setValue)
        self.levels_min_value.setToolTip("Manual contrast: intensity mapped to black")
        self.levels_max_value.setToolTip("Manual contrast: intensity mapped to white")
        self.follow_mouse = QCheckBox("Crosshair follows mouse")
        self.follow_mouse.setChecked(True)
        self.follow_mouse.toggled.connect(self._follow_mouse_changed)
        self.hist_chk = QCheckBox("Show histogram")
        self.hist_chk.setChecked(False)
        self.hist_chk.toggled.connect(self._histogram_changed)
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(ORIENTATIONS)
        self.orientation_combo.currentTextChanged.connect(self._orientation_changed)
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.setToolTip(
            "Fit the image to the viewport (does not move the crosshair)")
        self.reset_view_btn.clicked.connect(self._reset_view)

        def _level_row(slider, spin):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(slider, 1)
            row.addWidget(spin)
            box = QWidget()
            box.setLayout(row)
            return box

        disp_form.addRow("Palette:", self.palette_combo)
        disp_form.addRow("Pixel depth:", self.bits_combo)
        disp_form.addRow("Contrast:", self.contrast_auto)
        disp_form.addRow("Auto max:", self.auto_max)
        disp_form.addRow("Min:", _level_row(self.levels_min, self.levels_min_value))
        disp_form.addRow("Max:", _level_row(self.levels_max, self.levels_max_value))
        disp_form.addRow("Orientation:", self.orientation_combo)
        disp_form.addRow("", self.follow_mouse)
        disp_form.addRow("", self.hist_chk)
        disp_form.addRow("", self.reset_view_btn)
        right_lay.addWidget(disp_grp)

        # output location - shared by Save and Record, so neither needs a
        # modal file dialog any more
        out_grp = QGroupBox("Output location")
        out_form = QFormLayout(out_grp)
        self.out_dir_edit = QLineEdit(self._default_output_dir())
        self.out_dir_edit.setToolTip(
            "Folder that saved frames and recordings are written to")
        self.browse_dir_btn = QPushButton("Browse...")
        self.browse_dir_btn.setToolTip("Pick the output folder in the OS dialog")
        self.browse_dir_btn.clicked.connect(self._browse_output_dir)
        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.addWidget(self.out_dir_edit, 1)
        dir_row.addWidget(self.browse_dir_btn)
        dir_widget = QWidget()
        dir_widget.setLayout(dir_row)
        self.out_name_edit = QLineEdit("capture")
        self.out_name_edit.setToolTip(
            "Base file name, without extension: the selected format's suffix "
            "is added (.raw / .h5 / .zarr / .npy / .tiff, _seq for raw "
            "sequences). A typed image extension is replaced.")
        # Two lines from the start, so the panel's minimum height is known
        # when the scroll area is built (it is refreshed below).
        self.out_hint = QLabel("Save:\nRecord:")
        self.out_hint.setWordWrap(True)
        self.out_hint.setMaximumWidth(320)   # never widens the panel on long paths
        self.out_hint.setStyleSheet("color: gray;")
        out_form.addRow("Folder:", dir_widget)
        out_form.addRow("File name:", self.out_name_edit)
        out_form.addRow("", self.out_hint)
        right_lay.addWidget(out_grp)

        # save last frame
        save_grp = QGroupBox("Save last frame")
        save_form = QFormLayout(save_grp)
        self.save_fmt_combo = QComboBox()
        self.save_fmt_combo.addItems(["raw (binary + sidecar)", "HDF5 (+ sidecar)",
                                      "Zarr v3 + Blosc (+ sidecar)", "NumPy (.npy)",
                                      "BigTIFF (.tiff)"])
        self.blosc_cname_combo = QComboBox()
        self.blosc_cname_combo.addItems(["lz4", "zstd"])
        self.blosc_clevel_combo = QComboBox()
        self.blosc_clevel_combo.addItems(["1", "2", "3"])
        self.save_btn = QPushButton("Save last frame")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_last)
        save_form.addRow("Format:", self.save_fmt_combo)
        save_form.addRow("Blosc (Zarr):", self.blosc_cname_combo)
        save_form.addRow("Compression level:", self.blosc_clevel_combo)
        right_lay.addWidget(save_grp)

        # sequence record
        rec_grp = QGroupBox("Record sequence")
        rec_form = QFormLayout(rec_grp)
        self.rec_fmt_combo = QComboBox()
        self.rec_fmt_combo.addItems(["HDF5 (frames in one dataset)",
                                     "raw (folder of .raw + sidecar)",
                                     "Zarr (frames in one store)",
                                     "NumPy stack (.npy)",
                                     "BigTIFF stack (.tiff)"])
        self.rec_count = QSpinBox()
        self.rec_count.setRange(1, 100000)
        self.rec_count.setValue(100)
        self.rec_duration = QLabel("00:00")
        self.rec_count.valueChanged.connect(self._update_record_duration)
        self.rec_btn = QPushButton("Record N frames")
        self.rec_btn.setEnabled(False)
        self.rec_btn.clicked.connect(self._toggle_record)
        self.rec_progress = QProgressBar()
        self.rec_progress.setRange(0, 100)
        self.rec_progress.setValue(0)
        self.rec_progress.setFormat("Ready")
        self.rec_progress.setVisible(False)
        self.rec_progress.setAccessibleName("Recording progress")
        rec_form.addRow("Format:", self.rec_fmt_combo)
        rec_form.addRow("Frames:", self.rec_count)
        rec_form.addRow("Stack length (min:sec):", self.rec_duration)
        rec_form.addRow("Progress:", self.rec_progress)
        right_lay.addWidget(rec_grp)
        right_lay.addStretch(1)
        # Without an explicit minimum the scroll area sizes the column below
        # its own layout minimum, clipping the bottom of the last group.
        right.setMinimumHeight(right.minimumSizeHint().height())
        # The settings column is taller than a 1080p screen once every group is
        # open; let it scroll instead of forcing the whole window past the
        # bottom of the display.
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right)
        # Only vertical scrolling is wanted: keep the column wide enough for
        # its longest form row so no horizontal bar appears beside the image.
        right_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setMinimumWidth(right.minimumSizeHint().width() + 18)

        # The two actions are pinned below the scrolling settings, so Save and
        # Record stay clickable without scrolling whatever is configured above.
        panel = QWidget()
        panel_lay = QVBoxLayout(panel)
        panel_lay.setContentsMargins(4, 4, 4, 4)
        panel_lay.setSpacing(6)
        panel_lay.addWidget(right_scroll, 1)
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addWidget(self.save_btn, 1)
        actions.addWidget(self.rec_btn, 1)
        panel_lay.addLayout(actions)
        split.addWidget(panel)

        split.setSizes([430, 800, 722])
        split.setStretchFactor(1, 1)

        # log dock
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setWidget(self.log_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)
        self._log_dock.setFixedHeight(140)

        # status bar
        self.statusBar().addPermanentWidget(QLabel(BUILD_INFO))

        # menus
        menu_file = self.menuBar().addMenu("&File")
        act_raw = menu_file.addAction("Save last frame as raw", self._save_last_raw)
        act_raw.setShortcut("Ctrl+R")
        act_h5 = menu_file.addAction("Save last frame as HDF5", self._save_last_hdf5)
        act_h5.setShortcut("Ctrl+H")
        menu_file.addSeparator()
        menu_file.addAction("Save settings snapshot...", self._save_snapshot)
        menu_file.addSeparator()
        menu_file.addAction("Quit", self.close)

        menu_help = self.menuBar().addMenu("&Help")
        menu_help.addAction("About", self._about)

    def _connect_ui(self):
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.connect_btn.clicked.connect(self._connect_camera)
        self.disconnect_btn.clicked.connect(self._disconnect_camera)
        self.start_btn.clicked.connect(self._start_acq)
        self.stop_btn.clicked.connect(self._stop_acq)
        self.browser.feature_write.connect(self._on_feature_write)
        self.browser.feature_execute.connect(self._on_feature_execute)
        self.browser.refresh_requested.connect(
            lambda: self.worker is not None and
            self.worker.submit({"action": "refresh_tree"}))
        self.live.crosshairMoved.connect(self._on_crosshair)
        self.reader_tab.status.connect(self.log_line)
        # contrast controls (display only)
        self.contrast_auto.toggled.connect(self._contrast_mode_changed)
        self.auto_max.valueChanged.connect(self._apply_contrast)
        self.levels_min_value.valueChanged.connect(
            lambda _v: self._manual_level_edited("min"))
        self.levels_max_value.valueChanged.connect(
            lambda _v: self._manual_level_edited("max"))
        # output location preview
        self.out_dir_edit.textChanged.connect(self._update_output_hint)
        self.out_name_edit.textChanged.connect(self._update_output_hint)
        self.save_fmt_combo.currentIndexChanged.connect(self._update_output_hint)
        self.rec_fmt_combo.currentIndexChanged.connect(self._update_output_hint)
        self._update_level_ranges()
        self._contrast_mode_changed(self.contrast_auto.isChecked())
        self._update_output_hint()

    # ------------------------------------------------------------ logging
    def log_line(self, text: str):
        self.log_view.appendPlainText(time.strftime("%H:%M:%S") + "  " + text.strip())

    # ------------------------------------------------------- camera connect
    @pyqtSlot()
    def _source_changed(self):
        # The camera picker only applies to real hardware; switching back to
        # "Live" re-scans so a camera plugged in meanwhile shows up.
        self._sync_camera_picker()
        if not self._current_is_demo() and self.worker is None:
            self._refresh_cameras()

    def _sync_camera_picker(self):
        """The picker is usable only for live sources and only while disconnected."""
        enabled = (not self._current_is_demo()) and self.worker is None
        self.camera_combo.setEnabled(enabled)
        self.refresh_cameras_btn.setEnabled(enabled)

    @pyqtSlot()
    def _refresh_cameras(self):
        """Scan for connected FLIR cameras without blocking the UI."""
        if self._current_is_demo() or self.worker is not None:
            return
        if self._lister is not None and self._lister.isRunning():
            return                       # one scan at a time
        if not self._cameras and self.camera_combo.count() <= 1:
            self.camera_combo.setItemText(0, "Searching for cameras\u2026")
        self.refresh_cameras_btn.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self._lister = CameraLister(self)
        self._lister.camerasFound.connect(self._on_cameras_found)
        self._lister.failed.connect(self._on_cameras_failed)
        self._lister.finished.connect(self._sync_camera_picker)
        self._lister.start()

    def _on_cameras_found(self, cams):
        self._cameras = list(cams)
        # keep the current choice (or the --serial hint) selected across rescans
        wanted = self._wanted_serial or self.camera_combo.currentData()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        if not self._cameras:
            self.camera_combo.addItem("No cameras found - click Refresh")
        else:
            for cam in self._cameras:
                model = str(cam.get("model") or "FLIR camera").strip()
                serial = str(cam.get("serial") or "").strip()
                transport = str(cam.get("transport") or "").strip()
                label = f"{model}   SN={serial}" if serial else model
                if transport:
                    label = f"{label}  [{transport}]"
                self.camera_combo.addItem(label, serial or None)
        index = 0
        if wanted:
            for i in range(self.camera_combo.count()):
                if self.camera_combo.itemData(i) == wanted:
                    index = i
                    break
        self.camera_combo.setCurrentIndex(index)
        self.camera_combo.blockSignals(False)
        self._wanted_serial = None
        if self._cameras:
            self.log_line(f"Found {len(self._cameras)} camera(s).")
            self.statusBar().showMessage(
                f"{len(self._cameras)} camera(s) detected", 5000)
        else:
            self.log_line("No FLIR cameras detected.")

    def _on_cameras_failed(self, message: str):
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItem("Camera scan failed - click Refresh")
        self.camera_combo.blockSignals(False)
        self.log_line("ERROR: " + str(message))
        self.statusBar().showMessage(str(message), 8000)

    def _current_is_demo(self):
        text = self.source_combo.currentText()
        return text.startswith("Simulate:")

    def _current_model_key(self):
        text = self.source_combo.currentText()
        if text.startswith("Simulate:"):
            return text[len("Simulate: "):].strip()
        return None

    def _connect_camera(self):
        if self.worker is not None:
            return
        demo = self._current_is_demo()
        serial = None if demo else self.camera_combo.currentData()
        if serial:
            self.log_line(f"Connecting to camera SN={serial} \u2026")
        self.worker = create_worker(demo=demo, serial=serial,
                                    model_key=self._current_model_key())
        self.worker.frameReady.connect(self._on_frame)
        self.worker.treeReady.connect(self._on_tree)
        self.worker.rowUpdated.connect(self.browser.row_updated)
        self.worker.saveDone.connect(self._on_save_done)
        self.worker.stateChanged.connect(self._on_state)
        self.worker.error.connect(self._on_error)
        self.worker.log.connect(self.log_line)
        self.worker.set_display_settings(self.current_display_settings())
        self.worker.start()
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self._sync_camera_picker()   # lock the picker while connected
        self._refresh_timer.start()
        self._fps_timer.start()
        self._paint_timer.start()
        self._record_progress_timer.start()

    def _disconnect_camera(self):
        self._refresh_timer.stop()
        self._fps_timer.stop()
        self._paint_timer.stop()
        self._record_progress_timer.stop()
        self._pending_frame = None
        self._acq_fps = None
        self._frame_times.clear()
        if self.worker is not None:
            worker = self.worker
            worker.stop()
            worker.wait(5000)   # keep the reference until the thread has exited
            self.worker = None
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.rec_btn.setEnabled(False)
        self._recording = False
        self.rec_btn.setText("Record N frames")
        self.rec_progress.setVisible(False)
        self.live.clear_frame()
        self.lbl_state.setText("disconnected")
        self._sync_camera_picker()   # the picker may be used again

    # ------------------------------------------------------------ acq control
    def _start_acq(self):
        if self.worker:
            self._acq_fps = None
            self._frame_times.clear()
            self.worker.submit({"action": "start_acquisition"})

    def _stop_acq(self):
        if self.worker:
            self.worker.submit({"action": "stop_acquisition"})
        self._acq_fps = None
        self._frame_times.clear()

    # -------------------------------------------------------------- signals
    @pyqtSlot(dict)
    def _on_tree(self, tree: dict):
        self._last_tree = tree
        self.browser.set_tree(tree)
        self.log_line("Settings tree refreshed.")

    @pyqtSlot(str)
    def _on_state(self, state: str):
        self.lbl_state.setText(state)
        acquiring = state in ("acquiring", "recording")
        connected = state != "disconnected"
        self.start_btn.setEnabled(connected and not acquiring)
        self.stop_btn.setEnabled(connected and acquiring)
        self.save_btn.setEnabled(connected)
        self.rec_btn.setEnabled(connected)
        if state == "disconnected" and self.worker is None:
            pass
        elif state == "recording":
            self.rec_btn.setText("■ Stop recording")
            self._recording = True
        elif acquiring:
            self.rec_btn.setText("Record N frames" if not self._recording
                                 else "■ Stop recording")
        else:
            self._recording = False
            self.rec_btn.setText("Record N frames")

    def _refresh_record_progress(self):
        """Poll a small shared counter instead of queuing per-frame UI signals."""
        worker = self.worker
        progress_fn = getattr(worker, "recording_progress", None)
        progress = progress_fn() if callable(progress_fn) else None
        if progress is None:
            if not self._recording:
                self.rec_progress.setVisible(False)
            return
        total = int(progress["total"])
        captured = int(progress["captured"])
        remaining = int(progress["remaining"])
        status = str(progress.get("status") or "recording")
        if total <= 0:
            return
        self.rec_progress.setRange(0, total)
        self.rec_progress.setValue(min(captured, total))
        if status == "recording":
            self.rec_progress.setFormat(
                f"{max(0, remaining)} of {total} frames left")
        elif status == "saving":
            self.rec_progress.setFormat(
                f"Writing {captured} of {total} frames to disk \u2026")
        elif status == "complete":
            self.rec_progress.setFormat(f"Complete ({captured} frames)")
        else:
            self.rec_progress.setFormat("Recording failed")
        self.rec_progress.setVisible(True)

    @pyqtSlot(str)
    def _on_error(self, text: str):
        self.log_line("ERROR: " + text)
        self.statusBar().showMessage(text.splitlines()[0], 8000)

    @pyqtSlot(object, object)
    def _on_frame(self, arr, meta):
        # A frame posted just before the worker stopped can still be delivered
        # after a disconnect: drop anything that is not from the live worker.
        sender = self.sender()
        if sender is not None and sender is not self.worker:
            return
        source = sender if sender is not None else self.worker
        latest = source.take_latest_frame() if source is not None else None
        if latest is not None:
            # several frames can arrive while the GUI is busy: only the newest
            # one is ever painted, the rest are skipped for display only
            arr, meta = latest
        meta = dict(meta) if meta else {}
        self._pending_frame = (arr, meta)
        fps = meta.get("acquisition_fps")
        if fps:
            self._acq_fps = float(fps)
        self._frame_times.append(time.monotonic())
        if len(self._frame_times) > 240:
            self._frame_times = self._frame_times[-240:]
        self._frames_since_meta += 1
        if self._frames_since_meta % 8 == 0:
            self._update_info(meta)
        if source is None:
            self._paint_pending()    # no worker (offline pushes) - paint at once

    def _paint_pending(self):
        """Paint the newest frame, at most once per display interval."""
        frame, self._pending_frame = self._pending_frame, None
        if frame is None:
            return
        arr, meta = frame
        self.live.show_frame(arr, meta)

    def _update_fps(self):
        fps = self._acq_fps
        if fps is None:
            if not self._frame_times:
                self.lbl_fps.setText("0.0")
                self._update_record_duration()
                return
            span = max(0.001, self._frame_times[-1] - self._frame_times[0])
            n = len(self._frame_times) - 1
            fps = n / span if span > 0 else 0.0
        dropped = int(getattr(self.worker, "dropped_frames", 0) or 0)
        if self.live._arr is not None:
            hud = f"{fps:5.1f} fps"
            if dropped:
                hud += f"  {dropped} dropped"
            self.live.set_hud_text(hud)
        self.lbl_fps.setText(f"{fps:.1f}")
        self.lbl_fps.setToolTip(
            f"{dropped} frame(s) produced by the camera but never received by "
            f"the host; cap AcquisitionFrameRate below the host limit to "
            f"avoid drops." if dropped else "Acquisition frame rate (measured "
            "on the acquisition thread, independent of the display).")
        self._update_record_duration(fps)

    def _update_record_duration(self, fps=None):
        """Show requested stack duration using the current measured frame rate."""
        if fps is None:
            try:
                fps = float(self.lbl_fps.text())
            except (TypeError, ValueError):
                fps = 0.0
        seconds = int(self.rec_count.value() / fps) if fps and fps > 0 else 0
        minutes, seconds = divmod(seconds, 60)
        self.rec_duration.setText(f"{minutes:02d}:{seconds:02d}")

    def _follow_mouse_changed(self, follow: bool):
        self.live.set_follow_mouse(follow)
        self.reader_tab.view.set_follow_mouse(follow)

    def _histogram_changed(self, visible: bool):
        self.live.set_histogram_visible(visible)
        self.reader_tab.view.set_histogram_visible(visible)

    def _reset_view(self):
        """Fit whichever viewer the user is looking at."""
        widget = self.centre_tabs.currentWidget()
        if isinstance(widget, frame_reader.FrameReaderTab):
            widget.view.reset_view()
        else:
            self.live.reset_view()

    def _update_info(self, meta):
        if not meta:
            return
        self.lbl_frame.setText(str(meta.get("frame_id", "-")))
        self.lbl_dims.setText(f"{meta.get('width')} x {meta.get('height')}  "
                              f"{meta.get('acquired_dtype', '')}")
        self.lbl_format.setText(str(meta.get("pixel_format", "-")))

    @pyqtSlot(int, int, float)
    def _on_crosshair(self, x, y, val):
        self.lbl_cross.setText(f"x={x}  y={y}  val={val:.3f}")

    # --------------------------------------------------------- feature I/O
    def _on_feature_write(self, name: str, value):
        if self.worker:
            self.worker.submit({"action": "set_feature", "name": name, "value": value})

    def _on_feature_execute(self, name: str):
        if self.worker:
            self.worker.submit({"action": "execute_command", "name": name})

    def _poll_feature_values(self):
        if self.worker is not None:
            names = self.browser.collect_refresh_names()
            if names:
                self.worker.submit({"action": "refresh_rows", "names": names})

    # -------------------------------------------------------------- displays
    def current_display_settings(self):
        auto = self.contrast_auto.isChecked()
        return {
            "palette": self.palette_combo.currentText(),
            "follow_mouse": self.follow_mouse.isChecked(),
            "auto_levels": auto,
            "bit_depth": self._bits_value(),
            "auto_max_pct": float(self.auto_max.value()),
            "levels": None if auto else [self.levels_min_value.value(),
                                         self.levels_max_value.value()],
            "orientation": self.orientation_combo.currentText(),
        }

    def _bits_value(self) -> int:
        depth = self.bits_combo.currentData()
        try:
            return int(depth)
        except (TypeError, ValueError):
            return int(PixelBitDepth.TWELVE)

    def _level_top(self) -> int:
        return (1 << self._bits_value()) - 1

    def _level_controls(self):
        return (self.levels_min, self.levels_min_value,
                self.levels_max, self.levels_max_value)

    def _update_level_ranges(self):
        """Clamp the contrast sliders/spins to the selected pixel bitness."""
        top = self._level_top()
        old_top = self._levels_top
        widgets = self._level_controls()
        lo = self.levels_min_value.value()
        hi = self.levels_max_value.value()
        # If the handles covered the old full range, follow the new bitness
        # (8-bit 0-255 -> 12-bit 0-4095).  A custom window is only clamped.
        full_range = old_top is not None and lo <= 0 and hi >= old_top
        for w in widgets:
            w.blockSignals(True)
        try:
            for w in widgets:
                w.setMaximum(top)
            if full_range:
                lo, hi = 0, top
            else:
                lo = min(lo, top)
                hi = min(hi, top)
                if hi <= lo:
                    lo, hi = 0, top
            self._set_level_controls(lo, hi)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._levels_top = top
        self.levels_min_value.setToolTip(
            f"Manual contrast: intensity mapped to black (0-{top})")
        self.levels_max_value.setToolTip(
            f"Manual contrast: intensity mapped to white (0-{top})")

    def _set_level_controls(self, lo: int, hi: int):
        widgets = self._level_controls()
        for w in widgets:
            w.blockSignals(True)
        try:
            self.levels_min.setValue(lo)
            self.levels_max.setValue(hi)
            self.levels_min_value.setValue(lo)
            self.levels_max_value.setValue(hi)
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _manual_level_edited(self, which: str):
        """Keep min < max while the user edits either manual limit."""
        top = self._level_top()
        lo = self.levels_min_value.value()
        hi = self.levels_max_value.value()
        if lo >= hi:
            if which == "min":
                if lo < top:
                    hi = lo + 1
                else:
                    lo, hi = max(0, top - 1), top
            else:
                if hi > 0:
                    lo = hi - 1
                else:
                    lo, hi = 0, min(1, top)
        self._set_level_controls(lo, hi)
        self._apply_contrast()

    def _contrast_mode_changed(self, auto: bool):
        self.auto_max.setEnabled(bool(auto))
        for w in self._level_controls():
            w.setEnabled(not auto)
        self._apply_contrast()

    def _apply_contrast(self):
        """Push auto/manual contrast to both viewers (display only)."""
        settings = self.current_display_settings()
        self.live.set_bit_depth(settings["bit_depth"], settings["auto_levels"],
                                levels=settings["levels"],
                                auto_max_pct=settings["auto_max_pct"])
        self.reader_tab.set_contrast_settings(settings)
        if self.worker:
            self.worker.set_display_settings(settings)

    def _palette_changed(self, name):
        self._apply_display_settings()

    def _bits_changed(self, text):
        self._update_level_ranges()
        self._apply_display_settings()

    def _orientation_changed(self, name):
        self._apply_display_settings()

    def _apply_display_settings(self):
        """Apply the display controls to every viewer and the worker sidecar
        record."""
        settings = self.current_display_settings()
        self.live.set_palette(settings["palette"])
        self.live.set_bit_depth(settings["bit_depth"], settings["auto_levels"],
                                levels=settings["levels"],
                                auto_max_pct=settings["auto_max_pct"])
        self.live.set_orientation(settings["orientation"])
        self.reader_tab.set_display_settings(settings)
        if self.worker:
            self.worker.set_display_settings(settings)

    # ---------------------------------------------------------------- saving
    def _default_suggest(self, ext: str) -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return str(Path.home() / f"frame_{stamp}.{ext}")

    @staticmethod
    def _default_output_dir() -> str:
        location = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.PicturesLocation)
        return location or str(Path.home())

    def _browse_output_dir(self):
        """Pick the output folder with the native OS folder dialog."""
        start = self.out_dir_edit.text().strip() or self._default_output_dir()
        folder = QFileDialog.getExistingDirectory(
            self, "Select output folder", start)
        if folder:
            self.out_dir_edit.setText(folder)

    def _clean_output_base(self) -> Optional[Path]:
        """Folder / file name from the path fields, without a suffix.

        Returns None when either field is unusable, without talking to the
        user -- the preview label calls this on every keystroke.
        """
        folder_text = self.out_dir_edit.text().strip()
        name_text = self.out_name_edit.text().strip()
        if not folder_text or not name_text:
            return None
        folder = Path(folder_text).expanduser()
        if not folder.is_absolute():
            folder = Path.cwd() / folder
        # Only a file name may be typed here; directory parts are ignored so
        # "..\x" cannot escape the chosen folder.
        stem = Path(Path(name_text).name)
        if stem.suffix.lower() in _IMAGE_SUFFIXES:
            stem = stem.with_suffix("")
        if not stem.name or stem.name in (".", ".."):
            return None
        return folder / stem.name

    def _output_base(self) -> Optional[Path]:
        base = self._clean_output_base()
        if base is not None:
            return base
        if not self.out_dir_edit.text().strip():
            QMessageBox.warning(self, "Output folder missing",
                                "Set the output folder first (or use Browse...).")
        else:
            QMessageBox.warning(self, "File name missing",
                                "Set a file name for saved frames/recordings.")
        return None

    def _prepare_output(self, suffix: str) -> Optional[Path]:
        """Resolve folder/name + suffix, never overwriting existing data."""
        base = self._output_base()
        if base is None:
            return None
        path = Path(str(base) + suffix)
        if path.exists():
            existing = path.name
            for i in range(1, 10_000):
                candidate = Path(f"{base}_{i:03d}{suffix}")
                if not candidate.exists():
                    path = candidate
                    break
            else:
                QMessageBox.warning(self, "Output name busy",
                                    f"No free name next to {path}.")
                return None
            self.log_line(f"{existing} exists - writing {path.name} instead.")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, "Cannot create output folder",
                                f"{path.parent}\n\n{exc}")
            return None
        return path

    def _update_output_hint(self):
        base = self._clean_output_base()
        if base is None:
            self.out_hint.setText("Set a folder and a file name.")
            return
        save_fmt = ("raw", "hdf5", "zarr", "npy",
                    "tiff")[self.save_fmt_combo.currentIndex()]
        rec_fmt = ("hdf5_sequence", "raw_sequence", "zarr_sequence",
                   "npy_sequence", "tiff_sequence")[self.rec_fmt_combo.currentIndex()]
        # File names only: the folder is in the field right above, and a full
        # path is one unbreakable word that would stretch the whole panel.
        self.out_hint.setText(
            f"Save:   {Path(str(base) + _SAVE_SUFFIX[save_fmt]).name}\n"
            f"Record: {Path(str(base) + _RECORD_SUFFIX[rec_fmt]).name}")

    def _blosc_params(self):
        return {
            "blosc_cname": self.blosc_cname_combo.currentText(),
            "blosc_clevel": int(self.blosc_clevel_combo.currentText()),
            "blosc_shuffle": "shuffle",
        }

    def _save_last_hdf5(self):
        self._save_last(fmt="hdf5")

    def _save_last_raw(self):
        self._save_last(fmt="raw")

    def _save_last(self, fmt: Optional[str] = None):
        if self.worker is None:
            return
        # QPushButton.clicked passes its `checked` flag, which would arrive
        # here in place of *fmt* when the button is clicked.
        if not isinstance(fmt, str):
            fmt = None
        if fmt is None:
            fmt = ("raw", "hdf5", "zarr", "npy", "tiff")[self.save_fmt_combo.currentIndex()]
        path = self._prepare_output(_SAVE_SUFFIX[fmt])
        if path is None:
            return
        self.worker.submit({"action": "save_last", "path": str(path), "fmt": fmt,
                            "display_settings": self.current_display_settings(),
                            "blosc": self._blosc_params()})
        self.log_line(f"Saving frame to {path} \u2026")

    def _toggle_record(self):
        if self.worker is None:
            return
        if self._recording:
            self.worker.submit({"action": "stop_record"})
            return
        fmt = ("hdf5_sequence", "raw_sequence", "zarr_sequence",
               "npy_sequence", "tiff_sequence")[self.rec_fmt_combo.currentIndex()]
        n = int(self.rec_count.value())
        path = self._prepare_output(_RECORD_SUFFIX[fmt])
        if path is None:
            return
        self.worker.submit({
            "action": "record", "path": str(path), "n_frames": n, "fmt": fmt,
            "display_settings": self.current_display_settings(),
            "blosc": self._blosc_params()})
        self._recording = True
        self.rec_progress.setRange(0, n)
        self.rec_progress.setValue(0)
        self.rec_progress.setFormat(f"{n} of {n} frames left")
        self.rec_progress.setVisible(True)
        self.rec_btn.setText(f"■ Stop recording (rec {n})")

    def _save_snapshot(self):
        if not self._last_tree:
            QMessageBox.information(self, "Nothing to save",
                                    "Connect to a camera first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save camera settings snapshot", self._default_suggest("json"),
            "JSON (*.json)")
        if not path:
            return
        write_sidecar(Path(path), {
            "schema": 1, "application": "flir-cam-gui", "saved_at_utc":
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "camera_settings": self._last_tree})
        self.log_line(f"Settings snapshot written to {path}")

    @pyqtSlot(dict)
    def _on_save_done(self, result: dict):
        if result.get("ok"):
            self.log_line(result["message"])
            self.statusBar().showMessage(result["message"], 6000)
            if result.get("kind") == "recording":
                self._recording = False
                self.rec_btn.setText("Record N frames")
                progress_fn = getattr(self.worker, "recording_progress", None)
                progress = progress_fn() if callable(progress_fn) else None
                if progress and progress["total"]:
                    total = int(progress["total"])
                    captured = min(int(progress["captured"]), total)
                    self.rec_progress.setRange(0, total)
                    self.rec_progress.setValue(captured)
                    self.rec_progress.setFormat(f"Complete ({captured} frames)")
                self.rec_progress.setVisible(True)
        else:
            if result.get("kind") == "recording":
                self._recording = False
                self.rec_btn.setText("Record N frames")
                self.rec_progress.setFormat("Recording failed")
            else:
                self.rec_progress.setVisible(False)
            self.log_line("SAVE FAILED: " + result.get("message", ""))
            QMessageBox.warning(self, "Save failed", result.get("message", ""))

    def open_in_reader(self, path: str) -> bool:
        """Open *path* in the Saved images tab (also used by the CLI)."""
        self.centre_tabs.setCurrentWidget(self.reader_tab)
        self.reader_tab.set_display_settings(self.current_display_settings())
        return self.reader_tab.open_path(path)

    def _about(self):
        QMessageBox.information(
            self, "About", APP_TITLE + "\n"
            "PyQt6 + PySpin (FLIR Spinnaker) camera control.\n"
            "Simulation models are approximate, offline only.")

    # -------------------------------------------------------------- teardown
    def closeEvent(self, event):
        self._disconnect_camera()
        # a scan still in flight would be destroyed with the window, which Qt
        # turns into an abort; let it finish first
        lister = self._lister
        if lister is not None and lister.isRunning():
            lister.wait(5000)
        super().closeEvent(event)


BUILD_INFO = "flir-cam-gui 0.1.0"


def main():
    parser = argparse.ArgumentParser(description="FLIR Spinnaker camera GUI")
    parser.add_argument("--demo", action="store_true",
                        help="start in simulation mode even if PySpin is present")
    parser.add_argument("--live", action="store_true",
                        help="connect to real hardware even if PySpin appears missing")
    parser.add_argument("--model", type=str, default=None,
                        help="simulated camera model key (see README)")
    parser.add_argument("--serial", type=str, default=None, metavar="SN",
                        help="select this camera in the picker at startup")
    parser.add_argument("--open", type=str, default=None, metavar="PATH",
                        help="open a saved stack (raw/HDF5/Zarr) in the "
                             "Saved images tab at startup")
    args = parser.parse_args()
    # A fatal error inside PySpin/Qt aborts the interpreter without a Python
    # traceback; this prints the C-level stack so a crash is diagnosable
    # instead of silent.
    faulthandler.enable()
    app = QApplication(sys.argv)
    app.setApplicationName("flir-cam-gui")
    win = MainWindow(demo_default=args.demo, model_default=args.model,
                      serial_default=args.serial)
    win.show()
    # Always start disconnected - user must manually click Connect
    if args.open:
        win.open_in_reader(args.open)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
