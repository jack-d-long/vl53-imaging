#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


DATASET_OPTIONS = {
    "Signal (kcps)": "signal_kcps",
    "Ambient (kcps)": "ambient_kcps",
    "Distance (mm)": "distance_mm",
    "Status": "status",
    "Attempts": "attempts",
}


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel


def convolve_along_axis(data: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    finite = np.isfinite(data)
    values = np.where(finite, data, 0.0)
    weights = finite.astype(float)

    pad_spec = [(0, 0)] * data.ndim
    pad_spec[axis] = (pad, pad)

    values_padded = np.pad(values, pad_spec, mode="edge")
    weights_padded = np.pad(weights, pad_spec, mode="edge")

    accum = np.zeros_like(data, dtype=float)
    weight_accum = np.zeros_like(data, dtype=float)

    for index, weight in enumerate(kernel):
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(index, index + data.shape[axis])
        accum += weight * values_padded[tuple(slices)]
        weight_accum += weight * weights_padded[tuple(slices)]

    result = np.full_like(data, np.nan, dtype=float)
    np.divide(accum, weight_accum, out=result, where=weight_accum > 0)
    return result


def gaussian_blur_nan_safe(data: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return data.copy()
    kernel = gaussian_kernel_1d(sigma)
    blurred = convolve_along_axis(data, kernel, axis=1)
    blurred = convolve_along_axis(blurred, kernel, axis=0)
    return blurred


def upsample_grid(data: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return data.copy()
    return np.repeat(np.repeat(data, scale, axis=0), scale, axis=1)


class HeatmapCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(7, 6))
        super().__init__(self.figure)
        self.axes = self.figure.add_axes([0.08, 0.10, 0.72, 0.82])
        self.colorbar_axes = self.figure.add_axes([0.84, 0.10, 0.03, 0.82])
        self.colorbar = None
        self.image = None

    def render(
        self,
        grid: np.ndarray,
        title: str,
        cmap_name: str,
        interpolation: str,
        vmin: float | None,
        vmax: float | None,
    ) -> None:
        self.axes.clear()
        self.colorbar_axes.clear()
        cmap = matplotlib.colormaps[cmap_name].copy()
        cmap.set_bad(color="white")

        self.image = self.axes.imshow(
            grid,
            cmap=cmap,
            interpolation=interpolation,
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )
        self.axes.set_title(title)
        self.axes.set_xlabel("Grid column")
        self.axes.set_ylabel("Grid row")

        self.colorbar = self.figure.colorbar(self.image, cax=self.colorbar_axes)
        self.draw_idle()


class CaptureJsonViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VL53L1X Capture JSON Viewer")
        self.setAcceptDrops(True)

        self.current_path: Path | None = None
        self.current_payload: dict | None = None

        self.canvas = HeatmapCanvas()
        self.drop_label = QLabel("Drop a capture JSON here or click Open")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet("QLabel { border: 2px dashed #888; padding: 14px; }")

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(DATASET_OPTIONS.keys())
        self.dataset_combo.currentIndexChanged.connect(self.refresh_view)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "inferno",
            "viridis",
            "magma",
            "plasma",
            "cividis",
            "turbo",
            "gray",
        ])
        self.colormap_combo.currentTextChanged.connect(self.refresh_view)

        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["nearest", "bilinear", "bicubic"])
        self.interpolation_combo.currentTextChanged.connect(self.refresh_view)

        self.blur_sigma = QDoubleSpinBox()
        self.blur_sigma.setRange(0.0, 10.0)
        self.blur_sigma.setSingleStep(0.25)
        self.blur_sigma.setDecimals(2)
        self.blur_sigma.valueChanged.connect(self.refresh_view)

        self.render_scale = QSpinBox()
        self.render_scale.setRange(1, 64)
        self.render_scale.setValue(12)
        self.render_scale.valueChanged.connect(self.refresh_view)

        self.mask_invalid_checkbox = QCheckBox("Mask invalid zones")
        self.mask_invalid_checkbox.setChecked(True)
        self.mask_invalid_checkbox.stateChanged.connect(self.refresh_view)

        self.auto_range_checkbox = QCheckBox("Auto range")
        self.auto_range_checkbox.setChecked(True)
        self.auto_range_checkbox.stateChanged.connect(self.refresh_view)

        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.vmin_spin.setDecimals(3)
        self.vmin_spin.valueChanged.connect(self.refresh_view)

        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.vmax_spin.setDecimals(3)
        self.vmax_spin.setValue(1000.0)
        self.vmax_spin.valueChanged.connect(self.refresh_view)

        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setMinimumHeight(160)

        open_button = QPushButton("Open JSON")
        open_button.clicked.connect(self.open_json)

        reload_button = QPushButton("Reload")
        reload_button.clicked.connect(self.reload_current)

        button_row = QHBoxLayout()
        button_row.addWidget(open_button)
        button_row.addWidget(reload_button)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(self.drop_label)
        controls_layout.addLayout(button_row)

        form = QFormLayout()
        form.addRow("Dataset", self.dataset_combo)
        form.addRow("Colormap", self.colormap_combo)
        form.addRow("Interpolation", self.interpolation_combo)
        form.addRow("Render scale", self.render_scale)
        form.addRow("Gaussian sigma (cells)", self.blur_sigma)
        form.addRow(self.mask_invalid_checkbox)
        form.addRow(self.auto_range_checkbox)
        form.addRow("Color min", self.vmin_spin)
        form.addRow("Color max", self.vmax_spin)
        controls_layout.addLayout(form)
        controls_layout.addWidget(QLabel("Metadata"))
        controls_layout.addWidget(self.meta_text)
        controls_layout.addStretch(1)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.addWidget(self.canvas, stretch=3)
        layout.addWidget(controls, stretch=2)

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_json)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        self.menuBar().addAction(open_action)
        self.menuBar().addAction(quit_action)

        self.resize(1200, 720)
        self.update_range_controls()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(".json"):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() == ".json":
                self.load_capture(path)
                event.acceptProposedAction()
                return
        event.ignore()

    def open_json(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Open Capture JSON", "", "JSON Files (*.json)")
        if filename:
            self.load_capture(Path(filename))

    def reload_current(self) -> None:
        if self.current_path is not None:
            self.load_capture(self.current_path)

    def load_capture(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.drop_label.setText(f"Failed to load {path.name}: {exc}")
            return

        if payload.get("type") != "frame":
            self.drop_label.setText(f"{path.name} is not a frame capture JSON")
            return

        self.current_path = path
        self.current_payload = payload
        self.drop_label.setText(f"Loaded: {path}")
        self.populate_metadata(payload)
        self.refresh_view()

    def populate_metadata(self, payload: dict) -> None:
        lines = [
            f"File: {self.current_path}",
            f"Frame: {payload.get('frame')}",
            f"Rows x Cols: {payload.get('rows')} x {payload.get('cols')}",
            f"Zone count: {payload.get('zone_count')}",
            f"Frame time (ms): {payload.get('frame_time_ms')}",
        ]

        statuses = [zone.get("status") for zone in payload.get("zones", [])]
        invalid_count = sum(1 for status in statuses if status != 0)
        lines.append(f"Invalid zones: {invalid_count}")

        self.meta_text.setPlainText("\n".join(lines))

    def update_range_controls(self) -> None:
        manual = not self.auto_range_checkbox.isChecked()
        self.vmin_spin.setEnabled(manual)
        self.vmax_spin.setEnabled(manual)

    def selected_field(self) -> str:
        return DATASET_OPTIONS[self.dataset_combo.currentText()]

    def build_grid(self) -> np.ndarray | None:
        if self.current_payload is None:
            return None

        rows = int(self.current_payload["rows"])
        cols = int(self.current_payload["cols"])
        field = self.selected_field()
        mask_invalid = self.mask_invalid_checkbox.isChecked() and field not in {"status", "attempts"}

        grid = np.full((rows, cols), np.nan, dtype=float)
        for zone in self.current_payload["zones"]:
            row = zone["grid_row"]
            col = zone["grid_col"]
            if mask_invalid and zone.get("status", 1) != 0:
                continue
            value = zone.get(field)
            if value is None:
                continue
            grid[row, col] = float(value)

        sigma = self.blur_sigma.value()
        scale = self.render_scale.value()
        grid = upsample_grid(grid, scale)
        if sigma > 0:
            grid = gaussian_blur_nan_safe(grid, sigma * scale)

        return grid

    def auto_range(self, grid: np.ndarray) -> tuple[float | None, float | None]:
        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            return None, None
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if lo == hi:
            hi = lo + 1.0
        return lo, hi

    def refresh_view(self) -> None:
        self.update_range_controls()
        grid = self.build_grid()
        if grid is None:
            return

        if self.auto_range_checkbox.isChecked():
            vmin, vmax = self.auto_range(grid)
            if vmin is not None and vmax is not None:
                self.vmin_spin.blockSignals(True)
                self.vmax_spin.blockSignals(True)
                self.vmin_spin.setValue(vmin)
                self.vmax_spin.setValue(vmax)
                self.vmin_spin.blockSignals(False)
                self.vmax_spin.blockSignals(False)
        else:
            vmin = self.vmin_spin.value()
            vmax = self.vmax_spin.value()
            if vmin >= vmax:
                vmax = vmin + 1.0

        title = self.dataset_combo.currentText()
        if self.current_path is not None:
            title = f"{title} - {self.current_path.name}"

        self.canvas.render(
            grid=grid,
            title=title,
            cmap_name=self.colormap_combo.currentText(),
            interpolation=self.interpolation_combo.currentText(),
            vmin=vmin,
            vmax=vmax,
        )


def main() -> int:
    app = QApplication(sys.argv)
    viewer = CaptureJsonViewer()
    viewer.show()

    if len(sys.argv) > 1:
        viewer.load_capture(Path(sys.argv[1]))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
