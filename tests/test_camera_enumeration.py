"""Camera discovery must release Spinnaker handles in an order that cannot
abort the process.

PySpin raises ``SpinnakerException(-1004)`` when a camera handle is still
alive at ``CameraList.Clear()`` / ``System.ReleaseInstance()`` time, and the
half-released system then aborts the interpreter (STATUS_STACK_BUFFER_OVERRUN,
no Python traceback).  That is what made ``flir-cam`` die silently right after
startup.  The fake SDK below enforces the same rule, so these tests fail if
camera handles are kept past the release.
"""

import os
import weakref

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6", reason="PyQt6 required for the camera worker")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from flir_cam_gui import camera  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeSpinnakerError(Exception):
    """Stand-in for ``PySpin.SpinnakerException``."""


class _Node:
    def __init__(self, text: str):
        self._text = text

    def ToString(self) -> str:
        return self._text


class _NodeMap:
    def __init__(self, values):
        self._values = values

    def GetNode(self, name: str) -> _Node:
        return _Node(self._values.get(name, ""))


class _Camera:
    def __init__(self, sdk, values):
        self._sdk = sdk
        self._values = values
        self.inited = False

    def GetTLDeviceNodeMap(self) -> _NodeMap:
        return _NodeMap(self._values)

    def Init(self):
        self.inited = True

    def DeInit(self):
        self.inited = False


class _CameraList:
    def __init__(self, sdk, cameras):
        self._sdk = sdk
        self._cameras = list(cameras)

    def GetSize(self) -> int:
        return len(self._cameras)

    def GetByIndex(self, index: int) -> _Camera:
        cam = self._cameras[index]
        # real PySpin hands out a handle that keeps the camera referenced
        self._sdk.handles.append(weakref.ref(cam))
        return cam

    def Clear(self):
        self._sdk.clear_calls += 1
        self._cameras = []          # the list drops its own references ...
        self._sdk.assert_no_handles("Clear")   # ... external ones are fatal


class _System:
    def __init__(self, sdk):
        self._sdk = sdk

    def GetCameras(self) -> _CameraList:
        # like the SDK: the list owns a fresh set of camera objects
        return _CameraList(
            self._sdk, [_Camera(self._sdk, v) for v in self._sdk.values])

    def ReleaseInstance(self):
        self._sdk.release_calls += 1
        self._sdk.assert_no_handles("ReleaseInstance")


class _FakePySpin:
    """Minimal PySpin surface used by ``enumerate_cameras``/``_shutdown``."""

    def __init__(self, values_per_camera):
        self.values = list(values_per_camera)
        self.handles = []             # weakrefs to handed-out camera objects
        self.clear_calls = 0
        self.release_calls = 0
        self.system = _System(self)

    def new_camera(self, index: int = 0) -> _Camera:
        """A camera handle as the SDK would hand it to the caller."""
        cam = _Camera(self, self.values[index])
        self.handles.append(weakref.ref(cam))
        return cam

    # -- SDK surface ------------------------------------------------------
    class _SystemFactory:
        def __init__(self, sdk):
            self._sdk = sdk

        def GetInstance(self) -> _System:
            return self._sdk.system

    @property
    def System(self):                     # noqa: N802 - PySpin API name
        return _FakePySpin._SystemFactory(self)

    @staticmethod
    def CValuePtr(node) -> _Node:         # noqa: N802 - PySpin API name
        return node

    @staticmethod
    def IsReadable(node) -> bool:         # noqa: N802 - PySpin API name
        return bool(node.ToString())

    # -- assertions -------------------------------------------------------
    def live_handles(self):
        return [r() for r in self.handles if r() is not None]

    def assert_no_handles(self, where: str) -> None:
        if self.live_handles():
            raise _FakeSpinnakerError(
                f"Can't clear a camera because something still holds a "
                f"reference to the camera [-1004] (at {where})")


@pytest.fixture()
def fake_pyspin(monkeypatch):
    def install(cameras=None):
        fake = _FakePySpin(cameras if cameras is not None else [
            {"DeviceModelName": "Blackfly S BFS-U3-13Y3M",
             "DeviceSerialNumber": "19060813"},
            {"DeviceModelName": "Oryx ORX-10G",
             "DeviceSerialNumber": "21100001"},
        ])
        monkeypatch.setattr(camera, "PySpin", fake)
        monkeypatch.setattr(camera, "PYSPIN_AVAILABLE", True)
        return fake
    return install


def test_enumerate_cameras_reads_tl_device_nodes(fake_pyspin):
    fake_pyspin()
    cams = camera.enumerate_cameras()
    assert [c["model"] for c in cams] == [
        "Blackfly S BFS-U3-13Y3M", "Oryx ORX-10G"]
    assert [c["serial"] for c in cams] == ["19060813", "21100001"]


def test_enumerate_cameras_releases_after_dropping_camera_handles(fake_pyspin):
    fake = fake_pyspin()
    camera.enumerate_cameras()
    # the release only succeeds when no camera handle outlived the scan
    assert fake.clear_calls == 1
    assert fake.release_calls == 1
    assert not fake.live_handles()


def test_enumerate_cameras_is_repeatable(fake_pyspin):
    fake = fake_pyspin()
    camera.enumerate_cameras()
    camera.enumerate_cameras()
    assert fake.release_calls == 2
    assert not fake.live_handles()


def test_enumerate_cameras_without_pyspin(monkeypatch):
    monkeypatch.setattr(camera, "PYSPIN_AVAILABLE", False)
    assert camera.enumerate_cameras() == []


def test_enumerate_cameras_survives_release_errors(monkeypatch, fake_pyspin):
    fake = fake_pyspin()

    def boom():
        raise _FakeSpinnakerError("device busy")

    monkeypatch.setattr(fake.system, "ReleaseInstance", boom)
    # a failing release must not propagate out of discovery
    assert len(camera.enumerate_cameras()) == 2


def test_camera_lister_emits_scan_result(fake_pyspin, qapp):
    fake_pyspin()
    found = []
    lister = camera.CameraLister()
    lister.camerasFound.connect(found.append)
    lister.run()                      # the worker-thread entry point
    assert found and found[0][0]["serial"] == "19060813"


def test_worker_shutdown_releases_system(fake_pyspin, qapp):
    fake = fake_pyspin()
    worker = camera.CameraWorker()
    worker.cam = fake.new_camera(0)
    worker.nodemap = object()
    worker.stream_nodemap = object()
    worker._shutdown(fake.system.GetCameras(), fake.system)
    assert worker.cam is None and worker.nodemap is None
    assert fake.release_calls == 1
    assert not fake.live_handles()
