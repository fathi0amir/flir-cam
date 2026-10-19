"""Node browser: shows the full GenICam feature tree (camera + transport
layer node maps) and lets the user view/modify every available node.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QDoubleValidator, QIntValidator, QWheelEvent
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from . import camera as cam_defs

COL_NAME, COL_VALUE, COL_ACCESS = 0, 1, 2


class _NoWheelComboBox(QComboBox):
    """QComboBox that ignores wheel events so scrolling the tree never
    accidentally changes a value."""

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        event.ignore()


class NodeBrowser(QWidget):
    """Editable tree of every feature node on the camera."""

    feature_write = pyqtSignal(str, object)   # feature name, value
    feature_execute = pyqtSignal(str)         # command feature name
    refresh_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._rows: Dict[str, QTreeWidgetItem] = {}
        self._editors: Dict[str, QWidget] = {}
        self._updating = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter features...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        search_row.addWidget(self.search)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setMaximumWidth(90)
        refresh_btn.clicked.connect(self.refresh_requested.emit)
        search_row.addWidget(refresh_btn)
        lay.addLayout(search_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Feature", "Value", "Access"])
        self.tree.setColumnWidth(COL_NAME, 330)
        self.tree.setColumnWidth(COL_VALUE, 240)
        self.tree.setUniformRowHeights(True)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        lay.addWidget(self.tree, 1)

    # -- population -------------------------------------------------------------
    def set_tree(self, tree: Dict[str, Any]) -> None:
        self.tree.clear()
        self._rows.clear()
        self._editors.clear()
        for title, key in (("Camera Features", "camera"),
                           ("Transport Layer (Stream)", "stream")):
            root = tree.get(key)
            if not root:
                continue
            top = QTreeWidgetItem(self.tree, [title, "", ""])
            top.setFirstColumnSpanned(True)
            top.setBackground(0, self.tree.palette().button())
            top.setExpanded(True)
            for name, node in (root.get("features") or {}).items():
                self._populate(top, name, node)

    def _populate(self, parent: QTreeWidgetItem, name: str, node: Dict[str, Any],
                  path: str = "") -> None:
        if not isinstance(node, dict):
            return
        kind = node.get("__type__") or node.get("kind")
        display = node.get("display") or name
        if kind == cam_defs.KIND_CATEGORY:
            item = QTreeWidgetItem(parent, [display, "", ""])
            item.setData(0, Qt.ItemDataRole.UserRole, node)
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            for child_name, child in (node.get("features") or {}).items():
                self._populate(item, child_name, child, f"{path}{name}/")
            return
        row_key = f"{path}{name}"
        n_dupes = sum(1 for k in self._rows if k.split("#")[0] == row_key)
        if n_dupes:
            row_key = f"{row_key}#{n_dupes}"
        item = QTreeWidgetItem(parent, [display, "", self._access_str(node)])
        item.setData(0, Qt.ItemDataRole.UserRole, {**node, "name": name})
        item.setToolTip(0, f"{name}\n{node.get('tooltip') or ''}")
        self._rows[row_key] = item
        editor = self._make_editor(item, name, node, row_key)
        if editor is not None:
            self.tree.setItemWidget(item, COL_VALUE, editor)
            self._editors[row_key] = editor
        elif node.get("readable"):
            label = QLabel(self._fmt_value(node))
            label.setToolTip(item.toolTip(0))
            self.tree.setItemWidget(item, COL_VALUE, label)

    def _access_str(self, node: Dict[str, Any]) -> str:
        readable = bool(node.get("readable"))
        writable = bool(node.get("writable"))
        if readable and writable:
            return "RW"
        return "R" if readable else ("W" if writable else "-")

    def _make_editor(self, item: QTreeWidgetItem, name: str, node: Dict[str, Any],
                     row_key: str) -> Optional[QWidget]:
        kind = node.get("kind")
        if kind == cam_defs.KIND_COMMAND:
            btn = QPushButton("Execute ▶")
            btn.setMaximumHeight(18)
            btn.clicked.connect(lambda _, n=name: self.feature_execute.emit(n))
            return btn
        if not node.get("writable"):
            return None
        if kind == cam_defs.KIND_INTEGER:
            ln = QLineEdit()
            ln.setValidator(QIntValidator())
            ln.setText(str(node.get("value", "")) if node.get("value") is not None else "")
            ln.setToolTip(f"integer  min={node.get('min')} max={node.get('max')} "
                          f"inc={node.get('inc', 1)} {node.get('unit') or ''}")
            ln.editingFinished.connect(
                lambda: self._on_numeric_edit(ln, name, node, cast=int))
            return ln
        if kind == cam_defs.KIND_FLOAT:
            ln = QLineEdit()
            dv = QDoubleValidator()
            dv.setNotation(QDoubleValidator.Notation.StandardNotation)
            ln.setValidator(dv)
            ln.setText(self._fmt_value(node))
            ln.setToolTip(f"float  min={node.get('min')} max={node.get('max')} "
                          f"{node.get('unit') or ''}")
            ln.editingFinished.connect(
                lambda: self._on_numeric_edit(ln, name, node, cast=float))
            return ln
        if kind == cam_defs.KIND_BOOLEAN:
            cb = QCheckBox()
            cb.setChecked(bool(node.get("value")))
            cb.toggled.connect(lambda on, n=name: self._emit_write(n, bool(on)))
            return cb
        if kind == cam_defs.KIND_ENUMERATION:
            # Use _NoWheelComboBox so scrolling the tree doesn't accidentally
            # change an enum value when the mouse is hovering over the field.
            combo = _NoWheelComboBox()
            for entry in node.get("entries") or []:
                combo.addItem(entry.get("symbolic", str(entry.get("value"))))
            target = 0
            for i, entry in enumerate(node.get("entries") or []):
                if entry.get("current"):
                    target = i
            combo.setCurrentIndex(target)
            combo.currentTextChanged.connect(
                lambda sym, n=name: self._emit_write(n, sym))
            return combo
        if kind == cam_defs.KIND_STRING:
            ln = QLineEdit()
            ln.setText(str(node.get("value", "")))
            ln.editingFinished.connect(
                lambda: self._emit_write(name, ln.text()))
            return ln
        return None

    # -- editing behavior -------------------------------------------------------
    def _on_numeric_edit(self, ln: QLineEdit, name: str, node: Dict[str, Any], cast):
        """Commit a numeric edit. The editor text may carry the node's unit
        (e.g. '5000 µs'), which must never suppress the write: strip it and if
        the parse fails, show the *source* text so the user sees why.
        """
        if self._updating:
            return
        try:
            value = cast(self._strip_unit(ln.text(), node))
        except (TypeError, ValueError):
            ln.setText(self._fmt_value(node))
            return
        minimum, maximum = node.get("min"), node.get("max")
        if minimum is not None:
            value = max(cast(minimum), value)
        if maximum is not None:
            value = min(cast(maximum), value)
        self._emit_write(name, value)
        # Show the committed value with its unit immediately; the next
        # hardware readback replaces it with the authoritative value.
        if cast is float:
            ln.setText(self._fmt_value({**node, "value": value}))

    def _emit_write(self, name: str, value) -> None:
        # Programmatic row updates (e.g. a worker write-back) run inside the
        # _updating flag; never re-emit a write from those.
        if self._updating:
            return
        self.feature_write.emit(name, value)

    @staticmethod
    def _strip_unit(text: str, node: Dict[str, Any]) -> str:
        """Drop a unit suffix ('5000 µs' -> '5000') so the validators in the
        editor (which allow 'µs') never block a parseable value from reaching
        the camera."""
        numeric = text.strip()
        unit = (node.get("unit") or "").strip()
        if unit and numeric.lower().endswith(unit.lower()):
            numeric = numeric[: -len(unit)].strip()
        return numeric


    # -- row updates from the worker ---------------------------------------------
    def row_updated(self, info: Dict[str, Any]) -> None:
        if info.get("kind") in (cam_defs.KIND_CATEGORY,):
            return
        name = info.get("name") or ""
        for key, item in self._rows.items():
            node = item.data(0, Qt.ItemDataRole.UserRole) or {}
            if node.get("name") != name or node.get("kind") != info.get("kind"):
                continue
            self._row_update(key, item, info)

    def _row_update(self, key: str, item: QTreeWidgetItem, info: Dict[str, Any]) -> None:
        item.setText(COL_ACCESS, self._access_str(info))
        editor = self._editors.get(key)
        if editor is None:
            widget = self.tree.itemWidget(item, COL_VALUE)
            if isinstance(widget, QLabel) and info.get("readable"):
                widget.setText(self._fmt_value(info))
            return
        if editor.hasFocus():
            return
        self._updating = True
        try:
            if isinstance(editor, QLineEdit):
                editor.setText(self._fmt_value(info))
            elif isinstance(editor, QCheckBox):
                editor.setChecked(bool(info.get("value")))
            elif isinstance(editor, QComboBox):
                for entry in info.get("entries") or []:
                    if entry.get("current"):
                        if editor.currentText() != entry.get("symbolic"):
                            editor.setCurrentText(entry.get("symbolic"))
                        break
        finally:
            self._updating = False

    @staticmethod
    def _fmt_value(node: Dict[str, Any]) -> str:
        value = node.get("value")
        unit = node.get("unit") or ""
        if value is None:
            return ""
        if isinstance(value, float):
            value = f"{value:g}"
        return f"{value} {unit}".strip()

    # -- refresh / filter ---------------------------------------------------------
    def collect_refresh_names(self, limit: int = 300) -> List[str]:
        names: List[str] = []
        for key, item in self._rows.items():
            node = item.data(0, Qt.ItemDataRole.UserRole) or {}
            if node.get("readable") and node.get("kind") != cam_defs.KIND_COMMAND:
                plain = key.split("#")[0]
                if plain not in names:
                    names.append(plain)
            if len(names) >= limit:
                break
        return names

    def _apply_filter(self, text: str) -> None:
        text = text.strip().lower()

        def walk(item: QTreeWidgetItem) -> bool:
            matches = False
            for i in range(item.childCount()):
                if walk(item.child(i)):
                    matches = True
            own = text in item.text(COL_NAME).lower()
            visible = matches or own
            item.setHidden(bool(text) and not visible)
            return visible

        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))
