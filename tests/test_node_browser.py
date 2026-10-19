"""Node browser editors: numeric commits carry the unit, and programmatic
row updates never re-emit a write."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QLineEdit  # noqa: E402

from flir_cam_gui import camera as cam_defs  # noqa: E402
from flir_cam_gui.node_browser import NodeBrowser  # noqa: E402

# An ExposureTime-shaped node: float, unit suffix, min/max from the camera.
EXPOSURE = {
    "kind": cam_defs.KIND_FLOAT,
    "value": 5000.0,
    "unit": "µs",
    "min": 1.0,
    "max": 100000.0,
    "readable": True,
    "writable": True,
}

TREE = {
    "camera": {"__type__": cam_defs.KIND_CATEGORY, "display": "Root",
               "writable": False,
               "features": {"ExposureTime": dict(EXPOSURE)}},
    "stream": {"__type__": cam_defs.KIND_CATEGORY, "display": "Root (TLStream)",
               "writable": False, "features": {}},
}


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app



def _exposure_editor(b: NodeBrowser) -> QLineEdit:
    editors = [w for row, w in b._editors.items() if isinstance(w, QLineEdit)]
    assert len(editors) == 1
    return editors[0]


def test_float_editor_seeded_with_unit(qapp):
    b = NodeBrowser()
    b.set_tree(TREE)
    ln = _exposure_editor(b)
    assert ln.text() == "5000 µs"


def test_float_commit_with_unit_suffix_emits_write(qapp):
    b = NodeBrowser()
    writes = []
    b.feature_write.connect(lambda n, v: writes.append((n, v)))
    b.set_tree(TREE)
    ln = _exposure_editor(b)
    ln.setText("12500 µs")
    ln.editingFinished.emit()
    assert writes == [("ExposureTime", 12500.0)]
    # The displayed text keeps the unit and shows the committed value.
    assert ln.text() == "12500 µs"


def test_float_commit_still_accepts_plain_number(qapp):
    b = NodeBrowser()
    writes = []
    b.feature_write.connect(lambda n, v: writes.append((n, v)))
    b.set_tree(TREE)
    ln = _exposure_editor(b)
    ln.setText("7500")
    ln.editingFinished.emit()
    assert writes == [("ExposureTime", 7500.0)]


def test_float_commit_clamps_to_camera_limits(qapp):
    b = NodeBrowser()
    writes = []
    b.feature_write.connect(lambda n, v: writes.append((n, v)))
    b.set_tree(TREE)
    ln = _exposure_editor(b)
    ln.setText("999999 µs")
    ln.editingFinished.emit()
    assert writes == [("ExposureTime", 100000.0)]
    assert ln.text() == "100000 µs"


def test_float_unparseable_text_resets_and_does_not_write(qapp):
    b = NodeBrowser()
    writes = []
    b.feature_write.connect(lambda n, v: writes.append((n, v)))
    b.set_tree(TREE)
    ln = _exposure_editor(b)
    ln.setText("not a number")          # setText bypasses the validator
    ln.editingFinished.emit()
    assert writes == []
    assert ln.text() == "5000 µs"


def test_row_update_does_not_reemit_writes(qapp):
    b = NodeBrowser()
    writes = []
    b.feature_write.connect(lambda n, v: writes.append((n, v)))
    tree = {"camera": {"__type__": cam_defs.KIND_CATEGORY, "display": "Root",
                       "writable": False,
                       "features": {
                           "ExposureTime": dict(EXPOSURE),
                           "AcquisitionMode": {
                               "kind": cam_defs.KIND_ENUMERATION,
                               "value": "Continuous",
                               "entries": [
                                   {"symbolic": "Continuous",
                                    "value": 0, "current": True},
                                   {"symbolic": "SingleFrame", "value": 1},
                               ],
                               "readable": True, "writable": True}}},
            "stream": {"__type__": cam_defs.KIND_CATEGORY, "display": "Root",
                       "writable": False, "features": {}}}
    b.set_tree(tree)
    assert writes == []

    # A readback that changes the enum selection must not itself write.
    b.row_updated({"name": "AcquisitionMode",
                   "kind": cam_defs.KIND_ENUMERATION, "value": "SingleFrame",
                   "readable": True, "writable": True,
                   "entries": [{"symbolic": "Continuous", "value": 0},
                               {"symbolic": "SingleFrame", "value": 1,
                                "current": True}]})
    assert writes == []          # the update must not echo a write

    # A readback that changes the checkbox state must not itself write.
    tree2 = {"camera": {"__type__": cam_defs.KIND_CATEGORY, "display": "Root",
                        "writable": False,
                        "features": {"ReverseX": {
                            "kind": cam_defs.KIND_BOOLEAN, "value": False,
                            "readable": True, "writable": True}}},
             "stream": {"__type__": cam_defs.KIND_CATEGORY,
                        "display": "Root", "writable": False, "features": {}}}
    b.set_tree(tree2)
    b.row_updated({"name": "ReverseX", "kind": cam_defs.KIND_BOOLEAN,
                   "value": True, "readable": True, "writable": True})
    assert writes == []

    # ...but a genuine user edit still emits.
    b.row_updated({"name": "ReverseX", "kind": cam_defs.KIND_BOOLEAN,
                   "value": False, "readable": True, "writable": True})
    cb = next(w for w in b._editors.values())
    cb.setChecked(True)
    assert writes == [("ReverseX", True)]
