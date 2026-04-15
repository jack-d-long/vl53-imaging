#!/usr/bin/env python3

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import matplotlib
import numpy as np
import serial
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QAction, QCloseEvent, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
    "Composite (Signal + Ambient)": "signal_ambient_composite",
    "Distance (mm)": "distance_mm",
    "Status": "status",
    "Attempts": "attempts",
}
OVERLAY_OPTIONS = {
    "None": "none",
    "Signal": "signal",
    "Ambient": "ambient",
    "Signal + Ambient": "both",
}
INTERPOLATION_OPTIONS = ["linear", "nearest", "cubic", "idw", "gaussian"]
WEIGHTED_CHUNK_SIZE = 4096
COMPOSITE_FIELD = "signal_ambient_composite"
DEFAULT_CAPTURE_PATH = Path("latest_capture.json")
DEFAULT_SERIAL_PORT = "/dev/tty.usbserial-10"
DEFAULT_SERIAL_BAUD = 115200


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


def convolve_along_axis_raw(data: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    pad_spec = [(0, 0)] * data.ndim
    pad_spec[axis] = (pad, pad)
    padded = np.pad(data, pad_spec, mode="constant", constant_values=0.0)
    accum = np.zeros_like(data, dtype=float)

    for index, weight in enumerate(kernel):
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(index, index + data.shape[axis])
        accum += weight * padded[tuple(slices)]

    return accum


def gaussian_blur_nan_safe(data: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return data.copy()
    kernel = gaussian_kernel_1d(sigma)
    blurred = convolve_along_axis(data, kernel, axis=1)
    blurred = convolve_along_axis(blurred, kernel, axis=0)
    return blurred


def gaussian_blur_raw(data: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return data.copy()
    kernel = gaussian_kernel_1d(sigma)
    blurred = convolve_along_axis_raw(data, kernel, axis=1)
    blurred = convolve_along_axis_raw(blurred, kernel, axis=0)
    return blurred


def cubic_kernel(t: np.ndarray, a: float = -0.5) -> np.ndarray:
    abs_t = np.abs(t)
    abs_t2 = abs_t * abs_t
    abs_t3 = abs_t2 * abs_t
    result = np.zeros_like(abs_t, dtype=float)

    mask1 = abs_t <= 1.0
    result[mask1] = ((a + 2.0) * abs_t3[mask1]) - ((a + 3.0) * abs_t2[mask1]) + 1.0

    mask2 = (abs_t > 1.0) & (abs_t < 2.0)
    result[mask2] = (a * abs_t3[mask2]) - (5.0 * a * abs_t2[mask2]) + (8.0 * a * abs_t[mask2]) - (4.0 * a)
    return result


def grid_extent(x_coords: np.ndarray, y_coords: np.ndarray) -> tuple[float, float, float, float]:
    if x_coords.size > 1:
        dx = x_coords[1] - x_coords[0]
    else:
        dx = 1.0
    if y_coords.size > 1:
        dy = y_coords[1] - y_coords[0]
    else:
        dy = 1.0

    return (
        float(x_coords[0] - (dx / 2.0)),
        float(x_coords[-1] + (dx / 2.0)),
        float(y_coords[-1] + (dy / 2.0)),
        float(y_coords[0] - (dy / 2.0)),
    )


def background_value(field: str) -> float:
    if field in {"signal_kcps", "ambient_kcps", "attempts"}:
        return 0.0
    if field == "status":
        return 255.0
    if field == "distance_mm":
        return 0.0  # inverse-distance background corresponds to infinite distance
    return 0.0


def embed_in_background(grid: np.ndarray, field: str, lattice_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    rows, cols = grid.shape
    lattice_rows = max(lattice_size, rows)
    lattice_cols = max(lattice_size, cols)
    background = np.full((lattice_rows, lattice_cols), background_value(field), dtype=float)

    top = (lattice_rows - rows) // 2
    left = (lattice_cols - cols) // 2
    background[top : top + rows, left : left + cols] = grid
    return background, (top, left)


def transform_field(field: str, value: float) -> float:
    if field == "distance_mm":
        if value <= 0:
            return 0.0
        return 1.0 / value
    return value


def inverse_transform_field(field: str, grid: np.ndarray) -> np.ndarray:
    if field != "distance_mm":
        return grid

    result = np.full_like(grid, np.inf, dtype=float)
    valid = np.isfinite(grid) & (grid > 1e-12)
    result[valid] = 1.0 / grid[valid]
    return result


def build_render_domain(rows: int, cols: int, samples_per_cell: int) -> tuple[np.ndarray, np.ndarray]:
    x_coords = np.linspace(-0.5, cols - 0.5, cols * samples_per_cell)
    y_coords = np.linspace(-0.5, rows - 0.5, rows * samples_per_cell)
    return x_coords, y_coords


def extent_mask(
    shape: tuple[int, int],
    full_extent: tuple[float, float, float, float],
    target_extent: tuple[float, float, float, float],
) -> np.ndarray:
    rows, cols = shape
    full_x0, full_x1, full_y0, full_y1 = full_extent
    target_x0, target_x1, target_y0, target_y1 = target_extent

    x_coords = np.linspace(full_x0, full_x1, cols)
    y_coords = np.linspace(full_y1, full_y0, rows)
    x_mask = (x_coords >= target_x0) & (x_coords <= target_x1)
    y_mask = (y_coords >= target_y1) & (y_coords <= target_y0)
    return np.outer(y_mask, x_mask)


def interpolate_nearest(grid: np.ndarray, query_x: np.ndarray, query_y: np.ndarray) -> np.ndarray:
    rows, cols = grid.shape
    x = np.rint(query_x).astype(int)
    y = np.rint(query_y).astype(int)
    x = np.clip(x, 0, cols - 1)
    y = np.clip(y, 0, rows - 1)
    return grid[y, x]


def interpolate_idw(grid: np.ndarray, query_x: np.ndarray, query_y: np.ndarray, power: float = 2.0) -> np.ndarray:
    valid = np.isfinite(grid)
    if not np.any(valid):
        return np.full_like(query_x, np.nan, dtype=float)

    sample_y, sample_x = np.nonzero(valid)
    sample_values = grid[valid].astype(float)

    flat_x = query_x.ravel()
    flat_y = query_y.ravel()
    result = np.full(flat_x.shape[0], np.nan, dtype=float)

    for start in range(0, flat_x.shape[0], WEIGHTED_CHUNK_SIZE):
        stop = min(start + WEIGHTED_CHUNK_SIZE, flat_x.shape[0])
        chunk_x = flat_x[start:stop][None, :]
        chunk_y = flat_y[start:stop][None, :]

        distances2 = ((sample_x[:, None] - chunk_x) ** 2) + ((sample_y[:, None] - chunk_y) ** 2)
        exact = distances2 < 1e-12

        weights = 1.0 / np.maximum(distances2, 1e-12) ** (power / 2.0)
        weighted = (sample_values[:, None] * weights).sum(axis=0)
        weight_sum = weights.sum(axis=0)
        chunk_result = np.full(stop - start, np.nan, dtype=float)
        np.divide(weighted, weight_sum, out=chunk_result, where=weight_sum > 0)

        if np.any(exact):
            exact_columns = np.any(exact, axis=0)
            exact_indices = np.argmax(exact[:, exact_columns], axis=0)
            chunk_result[exact_columns] = sample_values[exact_indices]

        result[start:stop] = chunk_result

    return result.reshape(query_x.shape)


def interpolate_gaussian(grid: np.ndarray, query_x: np.ndarray, query_y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return interpolate_nearest(grid, query_x, query_y)

    valid = np.isfinite(grid)
    if not np.any(valid):
        return np.full_like(query_x, np.nan, dtype=float)

    sample_y, sample_x = np.nonzero(valid)
    sample_values = grid[valid].astype(float)

    flat_x = query_x.ravel()
    flat_y = query_y.ravel()
    result = np.full(flat_x.shape[0], np.nan, dtype=float)

    for start in range(0, flat_x.shape[0], WEIGHTED_CHUNK_SIZE):
        stop = min(start + WEIGHTED_CHUNK_SIZE, flat_x.shape[0])
        chunk_x = flat_x[start:stop][None, :]
        chunk_y = flat_y[start:stop][None, :]

        distances2 = ((sample_x[:, None] - chunk_x) ** 2) + ((sample_y[:, None] - chunk_y) ** 2)
        weights = np.exp(-0.5 * distances2 / (sigma * sigma))
        weighted = (sample_values[:, None] * weights).sum(axis=0)
        weight_sum = weights.sum(axis=0)
        np.divide(weighted, weight_sum, out=result[start:stop], where=weight_sum > 0)

    return result.reshape(query_x.shape)


def interpolate_idw_points(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_values: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    flat_x = query_x.ravel()
    flat_y = query_y.ravel()
    if sample_x.size == 0:
        return np.full_like(query_x, np.nan, dtype=float)

    result = np.full(flat_x.shape[0], np.nan, dtype=float)

    for start in range(0, flat_x.shape[0], WEIGHTED_CHUNK_SIZE):
        stop = min(start + WEIGHTED_CHUNK_SIZE, flat_x.shape[0])
        chunk_x = flat_x[start:stop][None, :]
        chunk_y = flat_y[start:stop][None, :]

        distances2 = ((sample_x[:, None] - chunk_x) ** 2) + ((sample_y[:, None] - chunk_y) ** 2)
        exact = distances2 < 1e-12

        weights = 1.0 / np.maximum(distances2, 1e-12) ** (power / 2.0)
        weighted = (sample_values[:, None] * weights).sum(axis=0)
        weight_sum = weights.sum(axis=0)
        chunk_result = np.full(stop - start, np.nan, dtype=float)
        np.divide(weighted, weight_sum, out=chunk_result, where=weight_sum > 0)

        if np.any(exact):
            exact_columns = np.any(exact, axis=0)
            exact_indices = np.argmax(exact[:, exact_columns], axis=0)
            chunk_result[exact_columns] = sample_values[exact_indices]

        result[start:stop] = chunk_result

    return result.reshape(query_x.shape)


def interpolate_gaussian_points(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_values: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    sigma: float,
) -> np.ndarray:
    if sigma <= 0:
        sigma = 1.0

    flat_x = query_x.ravel()
    flat_y = query_y.ravel()
    result = np.full(flat_x.shape[0], np.nan, dtype=float)

    for start in range(0, flat_x.shape[0], WEIGHTED_CHUNK_SIZE):
        stop = min(start + WEIGHTED_CHUNK_SIZE, flat_x.shape[0])
        chunk_x = flat_x[start:stop][None, :]
        chunk_y = flat_y[start:stop][None, :]

        distances2 = ((sample_x[:, None] - chunk_x) ** 2) + ((sample_y[:, None] - chunk_y) ** 2)
        weights = np.exp(-0.5 * distances2 / (sigma * sigma))
        weighted = (sample_values[:, None] * weights).sum(axis=0)
        weight_sum = weights.sum(axis=0)
        np.divide(weighted, weight_sum, out=result[start:stop], where=weight_sum > 0)

    return result.reshape(query_x.shape)


def extract_sample_points(sample_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(sample_grid)
    sample_y, sample_x = np.nonzero(valid)
    sample_values = sample_grid[valid].astype(float)
    return sample_x.astype(float), sample_y.astype(float), sample_values


def interpolate_bilinear(grid: np.ndarray, query_x: np.ndarray, query_y: np.ndarray) -> np.ndarray:
    rows, cols = grid.shape
    x = np.clip(query_x, 0.0, cols - 1.0)
    y = np.clip(query_y, 0.0, rows - 1.0)

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, cols - 1)
    y1 = np.clip(y0 + 1, 0, rows - 1)

    fx = x - x0
    fy = y - y0

    wx0 = np.where(x0 == x1, 1.0, 1.0 - fx)
    wx1 = np.where(x0 == x1, 0.0, fx)
    wy0 = np.where(y0 == y1, 1.0, 1.0 - fy)
    wy1 = np.where(y0 == y1, 0.0, fy)

    samples = [
        (grid[y0, x0], wx0 * wy0),
        (grid[y0, x1], wx1 * wy0),
        (grid[y1, x0], wx0 * wy1),
        (grid[y1, x1], wx1 * wy1),
    ]

    weighted = np.zeros_like(x, dtype=float)
    weight_sum = np.zeros_like(x, dtype=float)
    for values, weights in samples:
        valid = np.isfinite(values)
        weighted += np.where(valid, values, 0.0) * weights
        weight_sum += np.where(valid, weights, 0.0)

    result = np.full_like(x, np.nan, dtype=float)
    np.divide(weighted, weight_sum, out=result, where=weight_sum > 0)
    return result


def interpolate_bicubic(grid: np.ndarray, query_x: np.ndarray, query_y: np.ndarray) -> np.ndarray:
    return interpolate_bicubic_parametric(grid, query_x, query_y, a=-0.5)


def interpolate_bicubic_parametric(
    grid: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    a: float,
) -> np.ndarray:
    rows, cols = grid.shape
    x = np.clip(query_x, 0.0, cols - 1.0)
    y = np.clip(query_y, 0.0, rows - 1.0)

    base_x = np.floor(x).astype(int)
    base_y = np.floor(y).astype(int)
    frac_x = x - base_x
    frac_y = y - base_y

    x_offsets = np.array([-1, 0, 1, 2], dtype=int)
    y_offsets = np.array([-1, 0, 1, 2], dtype=int)

    x_indices = np.clip(base_x[..., None] + x_offsets, 0, cols - 1)
    y_indices = np.clip(base_y[..., None] + y_offsets, 0, rows - 1)

    x_weights = cubic_kernel(x_offsets.reshape((1,) * x.ndim + (4,)) - frac_x[..., None], a=a)
    y_weights = cubic_kernel(y_offsets.reshape((1,) * y.ndim + (4,)) - frac_y[..., None], a=a)

    weighted = np.zeros_like(x, dtype=float)
    weight_sum = np.zeros_like(x, dtype=float)

    for yi in range(4):
        for xi in range(4):
            values = grid[y_indices[..., yi], x_indices[..., xi]]
            weights = y_weights[..., yi] * x_weights[..., xi]
            valid = np.isfinite(values)
            weighted += np.where(valid, values, 0.0) * weights
            weight_sum += np.where(valid, weights, 0.0)

    result = np.full_like(x, np.nan, dtype=float)
    np.divide(weighted, weight_sum, out=result, where=np.abs(weight_sum) > 1e-12)
    return result


def build_sparse_background_anchors(
    real_grid: np.ndarray,
    field: str,
    lattice_shape: tuple[int, int],
    offset: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = real_grid.shape
    lattice_rows, lattice_cols = lattice_shape
    top, left = offset

    real_y, real_x = np.indices((rows, cols), dtype=float)
    real_y = real_y.ravel() + top
    real_x = real_x.ravel() + left
    real_values = real_grid.ravel()

    boundary_coords = []
    for x in range(lattice_cols):
        boundary_coords.append((0.0, float(x)))
        boundary_coords.append((float(lattice_rows - 1), float(x)))
    for y in range(1, lattice_rows - 1):
        boundary_coords.append((float(y), 0.0))
        boundary_coords.append((float(y), float(lattice_cols - 1)))

    boundary = np.array(boundary_coords, dtype=float)
    background = np.full(boundary.shape[0], background_value(field), dtype=float)

    sample_y = np.concatenate([real_y, boundary[:, 0]])
    sample_x = np.concatenate([real_x, boundary[:, 1]])
    sample_values = np.concatenate([real_values, background])
    return sample_x, sample_y, sample_values


def interpolate_grid(
    grid: np.ndarray,
    field: str,
    method: str,
    samples_per_cell: int,
    sigma: float,
    idw_power: float = 2.0,
    cubic_a: float = -0.5,
    real_grid: np.ndarray | None = None,
    real_offset: tuple[int, int] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    rows, cols = grid.shape
    x_coords, y_coords = build_render_domain(rows, cols, samples_per_cell)
    query_x, query_y = np.meshgrid(x_coords, y_coords)

    if method == "nearest":
        result = interpolate_nearest(grid, query_x, query_y)
    elif method == "linear":
        result = interpolate_bilinear(grid, query_x, query_y)
    elif method == "cubic":
        result = interpolate_bicubic_parametric(grid, query_x, query_y, a=cubic_a)
    elif method == "idw":
        if real_grid is not None and real_offset is not None:
            sample_x, sample_y, sample_values = build_sparse_background_anchors(real_grid, field, grid.shape, real_offset)
            result = interpolate_idw_points(sample_x, sample_y, sample_values, query_x, query_y, power=idw_power)
        else:
            result = interpolate_idw(grid, query_x, query_y, power=idw_power)
    elif method == "gaussian":
        if real_grid is not None and real_offset is not None:
            sample_x, sample_y, sample_values = build_sparse_background_anchors(real_grid, field, grid.shape, real_offset)
            result = interpolate_gaussian_points(sample_x, sample_y, sample_values, query_x, query_y, sigma=sigma)
        else:
            result = interpolate_gaussian(grid, query_x, query_y, sigma=sigma)
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")

    extent = (-0.5, cols - 0.5, rows - 0.5, -0.5)
    return inverse_transform_field(field, result), extent


def open_serial(port: str, baud: int) -> serial.Serial:
    try:
        return serial.Serial(port, baudrate=baud, timeout=0.05)
    except serial.SerialException as exc:
        raise RuntimeError(f"Failed to open {port}: {exc}") from exc


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
        vmin: float | None,
        vmax: float | None,
        extent: tuple[float, float, float, float] | None = None,
        view_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
        annotations: list[tuple[float, float, str]] | None = None,
    ) -> None:
        self.axes.clear()
        self.colorbar_axes.clear()
        cmap = matplotlib.colormaps[cmap_name].copy()
        cmap.set_bad(color="white")

        self.image = self.axes.imshow(
            grid,
            cmap=cmap,
            interpolation="nearest",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        self.axes.set_title(title)
        self.axes.set_xlabel("Grid column")
        self.axes.set_ylabel("Grid row")
        self.axes.set_aspect("equal")

        if annotations is not None:
            for x, y, text in annotations:
                self.axes.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                    family="monospace",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "black",
                        "alpha": 0.55,
                        "linewidth": 0,
                    },
                )

        if view_limits is not None:
            (x0, x1), (y0, y1) = view_limits
            self.axes.set_xlim(x0, x1)
            self.axes.set_ylim(y0, y1)

        self.colorbar = self.figure.colorbar(self.image, cax=self.colorbar_axes)
        self.draw_idle()

    def save_png(self, output_path: Path, dpi: int = 200) -> None:
        self.figure.savefig(output_path, dpi=dpi, bbox_inches="tight")


class CaptureJsonViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VL53L1X Capture JSON Viewer")
        self.setAcceptDrops(True)

        self.current_path: Path | None = None
        self.current_payload: dict | None = None
        self.current_grid_shape: tuple[int, int] | None = None
        self.last_valid_colormap = "gray"
        self.capture_path = DEFAULT_CAPTURE_PATH
        self.serial_connection: serial.Serial | None = None
        self.device_state = "disconnected"
        self.last_meta: dict | None = None
        self.serial_reader_thread: threading.Thread | None = None
        self.serial_stop_event = threading.Event()
        self.serial_message_queue: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self.serial_command_queue: queue.SimpleQueue[str] = queue.SimpleQueue()

        self.serial_timer = QTimer(self)
        self.serial_timer.setInterval(50)
        self.serial_timer.timeout.connect(self.process_serial_messages)

        self.canvas = HeatmapCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.drop_label = QLabel("Drop a capture JSON here or click Open")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet("QLabel { border: 2px dashed #888; padding: 14px; }")
        self.serial_status_label = QLabel("Serial: disconnected")
        self.serial_status_label.setWordWrap(True)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(DATASET_OPTIONS.keys())
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)

        self.serial_port_edit = QLineEdit()
        self.serial_port_edit.setPlaceholderText("/dev/ttyUSB0 or COM5")
        self.serial_port_edit.setText(DEFAULT_SERIAL_PORT)

        self.serial_baud_spin = QSpinBox()
        self.serial_baud_spin.setRange(1200, 4_000_000)
        self.serial_baud_spin.setSingleStep(1200)
        self.serial_baud_spin.setValue(DEFAULT_SERIAL_BAUD)

        self.capture_file_edit = QLineEdit(str(self.capture_path))
        self.capture_file_edit.editingFinished.connect(self.update_capture_path_from_ui)

        self.overlay_combo = QComboBox()
        self.overlay_combo.addItems(OVERLAY_OPTIONS.keys())
        self.overlay_combo.currentIndexChanged.connect(self.refresh_view)

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
        self.colormap_combo.setEditable(True)
        self.colormap_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.colormap_combo.setCurrentText("gray")
        self.colormap_combo.currentTextChanged.connect(self.refresh_view)

        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(INTERPOLATION_OPTIONS)
        self.interpolation_combo.setCurrentText("linear")
        self.interpolation_combo.currentTextChanged.connect(self.on_interpolation_changed)

        self.blur_sigma = QDoubleSpinBox()
        self.blur_sigma.setRange(0.0, 1000.0)
        self.blur_sigma.setSingleStep(0.25)
        self.blur_sigma.setDecimals(2)
        self.blur_sigma.setValue(0.0)
        self.blur_sigma.valueChanged.connect(self.refresh_view)

        self.idw_power = QDoubleSpinBox()
        self.idw_power.setRange(0.1, 16.0)
        self.idw_power.setSingleStep(0.25)
        self.idw_power.setDecimals(2)
        self.idw_power.setValue(2.0)
        self.idw_power.valueChanged.connect(self.refresh_view)

        self.cubic_a = QDoubleSpinBox()
        self.cubic_a.setRange(-2.0, 2.0)
        self.cubic_a.setSingleStep(0.05)
        self.cubic_a.setDecimals(2)
        self.cubic_a.setValue(-0.5)
        self.cubic_a.valueChanged.connect(self.refresh_view)

        self.render_scale = QSpinBox()
        self.render_scale.setRange(1, 64)
        self.render_scale.setValue(12)
        self.render_scale.valueChanged.connect(self.refresh_view)

        self.background_grid = QSpinBox()
        self.background_grid.setRange(13, 512)
        self.background_grid.setValue(20)
        self.background_grid.valueChanged.connect(self.refresh_view)

        self.composite_ambient_weight = QDoubleSpinBox()
        self.composite_ambient_weight.setRange(0.0, 1.0)
        self.composite_ambient_weight.setSingleStep(0.05)
        self.composite_ambient_weight.setDecimals(2)
        self.composite_ambient_weight.setValue(0.50)
        self.composite_ambient_weight.valueChanged.connect(self.refresh_view)

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

        export_button = QPushButton("Save PNG")
        export_button.clicked.connect(self.save_png)

        connect_button = QPushButton("Connect")
        connect_button.clicked.connect(self.connect_serial)

        disconnect_button = QPushButton("Disconnect")
        disconnect_button.clicked.connect(self.disconnect_serial)

        capture_button = QPushButton("Capture")
        capture_button.clicked.connect(self.capture_requested)

        read_capture_button = QPushButton("Read Capture File")
        read_capture_button.clicked.connect(self.read_capture_file)

        button_row = QHBoxLayout()
        button_row.addWidget(open_button)
        button_row.addWidget(reload_button)
        button_row.addWidget(export_button)

        serial_button_row = QHBoxLayout()
        serial_button_row.addWidget(connect_button)
        serial_button_row.addWidget(disconnect_button)
        serial_button_row.addWidget(capture_button)
        serial_button_row.addWidget(read_capture_button)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(self.drop_label)
        controls_layout.addLayout(button_row)
        controls_layout.addWidget(self.serial_status_label)
        controls_layout.addLayout(serial_button_row)
        controls_layout.addWidget(QLabel("Use the plot toolbar to zoom/pan, then Save PNG to export the current view."))

        form = QFormLayout()
        self.form = form
        form.addRow("Serial port", self.serial_port_edit)
        form.addRow("Serial baud", self.serial_baud_spin)
        form.addRow("Capture file", self.capture_file_edit)
        form.addRow("Dataset", self.dataset_combo)
        form.addRow("Ambient weight", self.composite_ambient_weight)
        form.addRow("Reading overlay", self.overlay_combo)
        form.addRow("Colormap", self.colormap_combo)
        form.addRow("Interpolation method", self.interpolation_combo)
        form.addRow("Background lattice", self.background_grid)
        form.addRow("Samples per cell", self.render_scale)
        form.addRow("Gaussian sigma", self.blur_sigma)
        form.addRow("IDW power", self.idw_power)
        form.addRow("Cubic a", self.cubic_a)
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

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        layout.addWidget(plot_panel, stretch=3)
        layout.addWidget(controls, stretch=2)

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_json)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        self.menuBar().addAction(open_action)
        self.menuBar().addAction(quit_action)

        self.resize(1200, 720)
        self.update_range_controls()
        self.on_interpolation_changed()
        self.on_dataset_changed()
        self.update_serial_status()

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

    def capture_file_path(self) -> Path:
        text = self.capture_file_edit.text().strip()
        if text:
            return Path(text)
        return DEFAULT_CAPTURE_PATH

    def update_capture_path_from_ui(self) -> None:
        self.capture_path = self.capture_file_path()
        self.capture_file_edit.setText(str(self.capture_path))
        self.update_serial_status()

    def update_serial_status(self, detail: str | None = None) -> None:
        target = self.capture_file_path()
        label = f"Serial: {self.device_state}"
        if detail:
            label = f"{label} | {detail}"
        label = f"{label} | capture file: {target}"
        self.serial_status_label.setText(label)

    def connect_serial(self) -> None:
        self.update_capture_path_from_ui()
        port = self.serial_port_edit.text().strip()
        if not port:
            self.device_state = "disconnected"
            self.update_serial_status("set a serial port first")
            return

        if self.serial_connection is not None:
            self.disconnect_serial()

        try:
            self.serial_connection = open_serial(port, self.serial_baud_spin.value())
        except RuntimeError as exc:
            self.device_state = "disconnected"
            self.update_serial_status(str(exc))
            return

        self.device_state = "connecting"
        self.last_meta = None
        self.serial_stop_event = threading.Event()
        self.serial_message_queue = queue.SimpleQueue()
        self.serial_command_queue = queue.SimpleQueue()
        self.serial_reader_thread = threading.Thread(
            target=self.serial_worker_main,
            name="capture-json-viewer-serial",
            daemon=True,
        )
        self.serial_reader_thread.start()
        self.serial_timer.start()
        self.update_serial_status(f"waiting for device reset on {port}")

    def serial_worker_main(self) -> None:
        assert self.serial_connection is not None

        ser = self.serial_connection
        buffer = bytearray()

        try:
            time.sleep(1.0)
            if self.serial_stop_event.is_set():
                return

            ser.reset_input_buffer()
            ser.write(b"meta\n")
            ser.flush()

            while not self.serial_stop_event.is_set():
                while True:
                    try:
                        command = self.serial_command_queue.get_nowait()
                    except queue.Empty:
                        break
                    ser.write((command + "\n").encode("utf-8"))
                    ser.flush()

                raw = ser.read(ser.in_waiting or 1)
                if not raw:
                    continue

                buffer.extend(raw)
                while b"\n" in buffer:
                    line_bytes, _, remainder = buffer.partition(b"\n")
                    buffer = bytearray(remainder)
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        self.serial_message_queue.put(("raw", line))
                        continue

                    self.serial_message_queue.put(("payload", payload))
        except serial.SerialException as exc:
            self.serial_message_queue.put(("error", f"serial error: {exc}"))
        finally:
            try:
                ser.close()
            except serial.SerialException:
                pass
            self.serial_message_queue.put(("closed", None))

    def disconnect_serial(self) -> None:
        self.serial_timer.stop()
        self.serial_stop_event.set()
        if self.serial_connection is not None:
            try:
                self.serial_connection.close()
            except serial.SerialException:
                pass
        if self.serial_reader_thread is not None:
            self.serial_reader_thread.join(timeout=0.5)
        self.serial_reader_thread = None
        self.serial_connection = None
        self.serial_message_queue = queue.SimpleQueue()
        self.serial_command_queue = queue.SimpleQueue()
        self.device_state = "disconnected"
        self.update_serial_status()

    def send_serial_command(self, command: str) -> None:
        if self.serial_connection is None:
            self.update_serial_status("not connected")
            return
        self.serial_command_queue.put(command)

    def capture_requested(self) -> None:
        if self.serial_connection is None:
            self.update_serial_status("not connected")
            return
        if self.device_state == "connecting":
            self.update_serial_status("device is still resetting")
            return
        self.device_state = "requesting"
        self.update_serial_status("capture requested")
        self.send_serial_command("capture")

    def read_capture_file(self) -> None:
        self.update_capture_path_from_ui()
        if not self.capture_path.exists():
            self.update_serial_status(f"no capture file: {self.capture_path.name}")
            return
        self.load_capture(self.capture_path)

    def save_png(self) -> None:
        if self.current_payload is None or self.current_path is None:
            self.drop_label.setText("Load a capture JSON before saving PNG")
            return

        default_name = self.current_path.with_suffix(".png").name
        filename, _ = QFileDialog.getSaveFileName(self, "Save PNG", default_name, "PNG Files (*.png)")
        if not filename:
            return

        self.canvas.save_png(Path(filename))
        self.drop_label.setText(f"Saved PNG: {filename}")

    def reload_current(self) -> None:
        if self.current_path is not None:
            self.load_capture(self.current_path)

    def apply_frame_payload(self, payload: dict, source_path: Path | None = None) -> None:
        if payload.get("type") != "frame":
            self.drop_label.setText("JSON payload is not a frame capture")
            return

        if source_path is not None:
            self.current_path = source_path
            self.capture_path = source_path
            self.capture_file_edit.setText(str(source_path))
        self.current_payload = payload
        self.populate_metadata(payload)
        self.refresh_view()

    def load_capture(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.drop_label.setText(f"Failed to load {path.name}: {exc}")
            return

        if payload.get("type") != "frame":
            self.drop_label.setText(f"{path.name} is not a frame capture JSON")
            return

        self.drop_label.setText(f"Loaded: {path}")
        self.apply_frame_payload(payload, source_path=path)

    def populate_metadata(self, payload: dict) -> None:
        lines = [
            f"File: {self.current_path}",
            f"Frame: {payload.get('frame')}",
            f"Rows x Cols: {payload.get('rows')} x {payload.get('cols')}",
            f"Zone count: {payload.get('zone_count')}",
            f"Frame time (ms): {payload.get('frame_time_ms')}",
        ]

        if self.last_meta is not None:
            lines.append(
                "Meta rows x cols: {rows} x {cols}".format(
                    rows=self.last_meta.get("rows"),
                    cols=self.last_meta.get("cols"),
                )
            )

        statuses = [zone.get("status") for zone in payload.get("zones", [])]
        invalid_count = sum(1 for status in statuses if status != 0)
        lines.append(f"Invalid zones: {invalid_count}")

        self.meta_text.setPlainText("\n".join(lines))

    def save_frame(self, payload: dict) -> None:
        self.update_capture_path_from_ui()
        self.capture_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def handle_serial_meta(self, payload: dict) -> None:
        self.last_meta = payload
        self.device_state = "idle"
        self.update_serial_status("metadata received")
        if self.current_payload is not None:
            self.populate_metadata(self.current_payload)

    def handle_serial_state(self, payload: dict) -> None:
        state = payload.get("state")
        if isinstance(state, str):
            self.device_state = state
            if state == "idle":
                self.update_serial_status("ready")
            elif state == "capturing":
                self.update_serial_status("capturing")
            else:
                self.update_serial_status("device update")

    def handle_serial_frame(self, payload: dict) -> None:
        try:
            self.save_frame(payload)
        except OSError as exc:
            self.update_serial_status(f"failed to save frame: {exc}")
            return
        self.device_state = "idle"
        self.drop_label.setText(f"Captured: {self.capture_path}")
        self.apply_frame_payload(payload, source_path=self.capture_path)
        frame_id = payload.get("frame")
        frame_time_ms = payload.get("frame_time_ms")
        self.update_serial_status(f"frame {frame_id} | {frame_time_ms} ms")

    def handle_serial_payload(self, payload: dict) -> None:
        message_type = payload.get("type")
        if message_type == "meta":
            self.handle_serial_meta(payload)
        elif message_type == "state":
            self.handle_serial_state(payload)
        elif message_type == "frame":
            self.handle_serial_frame(payload)
        elif message_type == "help":
            self.update_serial_status("device help received")
        elif message_type == "error":
            self.update_serial_status(str(payload.get("message", "device error")))

    def process_serial_messages(self) -> None:
        while True:
            try:
                kind, value = self.serial_message_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "payload":
                assert isinstance(value, dict)
                self.handle_serial_payload(value)
            elif kind == "raw":
                print(f"Skipping non-JSON line: {value}", file=sys.stderr)
            elif kind == "error":
                self.disconnect_serial()
                self.update_serial_status(str(value))
                break
            elif kind == "closed":
                if self.serial_stop_event.is_set():
                    break
                self.disconnect_serial()
                self.update_serial_status("serial connection closed")
                break

    def update_range_controls(self) -> None:
        manual = not self.auto_range_checkbox.isChecked()
        self.vmin_spin.setEnabled(manual)
        self.vmax_spin.setEnabled(manual)

    def set_form_row_visible(self, field_widget: QWidget, visible: bool) -> None:
        label = self.form.labelForField(field_widget)
        if label is not None:
            label.setVisible(visible)
        field_widget.setVisible(visible)

    def on_interpolation_changed(self) -> None:
        method = self.interpolation_combo.currentText()
        uses_background = method in {"linear", "nearest", "cubic", "gaussian"}
        uses_sigma = method == "gaussian"
        uses_idw_power = method == "idw"
        uses_cubic_a = method == "cubic"

        self.set_form_row_visible(self.background_grid, uses_background)
        self.set_form_row_visible(self.blur_sigma, uses_sigma)
        self.set_form_row_visible(self.idw_power, uses_idw_power)
        self.set_form_row_visible(self.cubic_a, uses_cubic_a)
        self.refresh_view()

    def selected_field(self) -> str:
        return DATASET_OPTIONS[self.dataset_combo.currentText()]

    def on_dataset_changed(self) -> None:
        self.set_form_row_visible(self.composite_ambient_weight, self.selected_field() == COMPOSITE_FIELD)
        self.refresh_view()

    def selected_overlay_mode(self) -> str:
        return OVERLAY_OPTIONS[self.overlay_combo.currentText()]

    def current_colormap(self) -> str:
        cmap_name = self.colormap_combo.currentText().strip()
        if cmap_name in matplotlib.colormaps:
            self.last_valid_colormap = cmap_name
            return cmap_name
        return self.last_valid_colormap

    def build_field_sample_grids(self, field: str) -> tuple[np.ndarray, np.ndarray]:
        assert self.current_payload is not None

        rows = int(self.current_payload["rows"])
        cols = int(self.current_payload["cols"])
        mask_invalid = self.mask_invalid_checkbox.isChecked()
        fill_value = background_value(field)
        sample_grid = np.full((rows, cols), np.nan, dtype=float)
        background_grid = np.full((rows, cols), fill_value, dtype=float)

        for zone in self.current_payload["zones"]:
            row = zone["grid_row"]
            col = zone["grid_col"]
            if mask_invalid and zone.get("status", 1) != 0:
                continue
            value = zone.get(field)
            if value is None:
                continue
            transformed = transform_field(field, float(value))
            sample_grid[row, col] = transformed
            background_grid[row, col] = transformed

        return sample_grid, background_grid

    def build_overlay_annotations(self, x_offset: float, y_offset: float) -> list[tuple[float, float, str]]:
        if self.current_payload is None:
            return []

        overlay_mode = self.selected_overlay_mode()
        if overlay_mode == "none":
            return []

        mask_invalid = self.mask_invalid_checkbox.isChecked()
        annotations: list[tuple[float, float, str]] = []
        for zone in self.current_payload["zones"]:
            if mask_invalid and zone.get("status", 1) != 0:
                continue

            parts = []
            if overlay_mode in {"signal", "both"}:
                signal = zone.get("signal_kcps")
                if signal is not None:
                    parts.append(f"S:{int(signal)}")
            if overlay_mode in {"ambient", "both"}:
                ambient = zone.get("ambient_kcps")
                if ambient is not None:
                    parts.append(f"A:{int(ambient)}")

            if not parts:
                continue

            annotations.append(
                (
                    float(zone["grid_col"]) + x_offset,
                    float(zone["grid_row"]) + y_offset,
                    "\n".join(parts),
                )
            )

        return annotations

    def build_interpolated_grid(
        self,
        field: str,
    ) -> tuple[
        np.ndarray,
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        list[tuple[float, float, str]],
    ]:
        assert self.current_payload is not None

        rows = int(self.current_payload["rows"])
        cols = int(self.current_payload["cols"])
        sample_grid, background_grid = self.build_field_sample_grids(field)
        method = self.interpolation_combo.currentText()

        if method == "idw":
            sample_x, sample_y, sample_values = extract_sample_points(sample_grid)
            x_coords, y_coords = build_render_domain(rows, cols, self.render_scale.value())
            query_x, query_y = np.meshgrid(x_coords, y_coords)
            interpolated = interpolate_idw_points(
                sample_x=sample_x,
                sample_y=sample_y,
                sample_values=sample_values,
                query_x=query_x,
                query_y=query_y,
                power=self.idw_power.value(),
            )
            extent = (-0.5, cols - 0.5, rows - 0.5, -0.5)
            interpolated = inverse_transform_field(field, interpolated)
            sampled_extent = extent
            annotations = self.build_overlay_annotations(x_offset=0.0, y_offset=0.0)
        else:
            lattice, (top, left) = embed_in_background(background_grid, field, self.background_grid.value())

            interpolated, extent = interpolate_grid(
                grid=lattice,
                field=field,
                method=method,
                samples_per_cell=self.render_scale.value(),
                sigma=self.blur_sigma.value(),
                cubic_a=self.cubic_a.value(),
                real_grid=sample_grid,
                real_offset=(top, left),
            )
            sampled_extent = (
                left - 0.5,
                left + cols - 0.5,
                top + rows - 0.5,
                top - 0.5,
            )
            annotations = self.build_overlay_annotations(x_offset=float(left), y_offset=float(top))

        return interpolated, extent, sampled_extent, annotations

    def normalize_grid_for_composite(
        self,
        grid: np.ndarray,
        full_extent: tuple[float, float, float, float],
        sampled_extent: tuple[float, float, float, float],
    ) -> np.ndarray:
        sampled_mask = extent_mask(grid.shape, full_extent, sampled_extent)
        finite_sampled = grid[np.isfinite(grid) & sampled_mask]
        normalized = np.full_like(grid, np.nan, dtype=float)
        if finite_sampled.size == 0:
            return normalized

        lo = float(np.nanmin(finite_sampled))
        hi = float(np.nanmax(finite_sampled))
        finite = np.isfinite(grid)
        if hi <= lo + 1e-12:
            normalized[finite] = 1.0 if hi > 0.0 else 0.0
            return normalized

        normalized[finite] = np.clip((grid[finite] - lo) / (hi - lo), 0.0, 1.0)
        return normalized

    def build_composite_grid(
        self,
    ) -> tuple[
        np.ndarray,
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        list[tuple[float, float, str]],
    ]:
        signal_grid, extent, sampled_extent, annotations = self.build_interpolated_grid("signal_kcps")
        ambient_grid, _, _, _ = self.build_interpolated_grid("ambient_kcps")

        signal_normalized = self.normalize_grid_for_composite(signal_grid, extent, sampled_extent)
        ambient_normalized = self.normalize_grid_for_composite(ambient_grid, extent, sampled_extent)

        ambient_weight = float(self.composite_ambient_weight.value())
        signal_weight = 1.0 - ambient_weight

        weighted_sum = (
            np.where(np.isfinite(signal_normalized), signal_normalized * signal_weight, 0.0)
            + np.where(np.isfinite(ambient_normalized), ambient_normalized * ambient_weight, 0.0)
        )
        weight_sum = (
            np.where(np.isfinite(signal_normalized), signal_weight, 0.0)
            + np.where(np.isfinite(ambient_normalized), ambient_weight, 0.0)
        )
        composite = np.full_like(signal_normalized, np.nan, dtype=float)
        np.divide(weighted_sum, weight_sum, out=composite, where=weight_sum > 0)
        return composite, extent, sampled_extent, annotations

    def build_grid(
        self,
    ) -> tuple[
        np.ndarray,
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        list[tuple[float, float, str]],
    ] | None:
        if self.current_payload is None:
            return None

        field = self.selected_field()
        if field == COMPOSITE_FIELD:
            return self.build_composite_grid()
        return self.build_interpolated_grid(field)

    def auto_range(
        self,
        grid: np.ndarray,
        full_extent: tuple[float, float, float, float],
        sampled_extent: tuple[float, float, float, float],
    ) -> tuple[float | None, float | None]:
        sampled_mask = extent_mask(grid.shape, full_extent, sampled_extent)
        finite = grid[np.isfinite(grid) & sampled_mask]
        if finite.size == 0:
            return None, None
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if lo == hi:
            hi = lo + 1.0
        return lo, hi

    def current_view_limits(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if self.canvas.image is None:
            return None

        xlim = self.canvas.axes.get_xlim()
        ylim = self.canvas.axes.get_ylim()
        default_xlim = (self.canvas.image.get_extent()[0], self.canvas.image.get_extent()[1])
        default_ylim = (self.canvas.image.get_extent()[2], self.canvas.image.get_extent()[3])
        if np.allclose(xlim, default_xlim) and np.allclose(ylim, default_ylim):
            return None

        return ((float(xlim[0]), float(xlim[1])), (float(ylim[0]), float(ylim[1])))

    def refresh_view(self) -> None:
        self.update_range_controls()
        existing_limits = self.current_view_limits()
        built = self.build_grid()
        if built is None:
            return
        grid, extent, sampled_extent, annotations = built
        self.current_grid_shape = grid.shape

        if self.auto_range_checkbox.isChecked():
            vmin, vmax = self.auto_range(grid, extent, sampled_extent)
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
        if self.selected_field() == COMPOSITE_FIELD:
            ambient_weight = self.composite_ambient_weight.value()
            signal_weight = 1.0 - ambient_weight
            title = f"{title} [signal={signal_weight:.2f}, ambient={ambient_weight:.2f}]"
        if self.current_path is not None:
            title = f"{title} - {self.current_path.name}"

        typed_colormap = self.colormap_combo.currentText().strip()
        if typed_colormap and typed_colormap not in matplotlib.colormaps:
            title = f"{title}  |  invalid colormap: {typed_colormap}"

        self.canvas.render(
            grid=grid,
            title=title,
            cmap_name=self.current_colormap(),
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            view_limits=existing_limits,
            annotations=annotations,
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.disconnect_serial()
        super().closeEvent(event)


def main() -> int:
    parser = argparse.ArgumentParser(description="VL53L1X capture JSON viewer / PNG exporter")
    parser.add_argument("capture_json", nargs="?", help="Optional capture JSON to load")
    parser.add_argument("--export-png", help="Export directly to PNG without launching the UI")
    parser.add_argument(
        "--port",
        default=DEFAULT_SERIAL_PORT,
        help="Serial port, for example /dev/ttyUSB0 or COM5",
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_SERIAL_BAUD, help="Serial baud rate")
    parser.add_argument("--capture-file", default=str(DEFAULT_CAPTURE_PATH), help="Path for captured frame JSON")
    parser.add_argument("--auto-connect", action="store_true", help="Connect to the serial port on launch")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_OPTIONS.keys()),
        default="Signal (kcps)",
        help="Dataset to render for export",
    )
    parser.add_argument(
        "--overlay-readings",
        choices=list(OVERLAY_OPTIONS.keys()),
        default="None",
        help="Optional per-zone reading overlay for export",
    )
    parser.add_argument("--colormap", default="gray", help="Matplotlib colormap name")
    parser.add_argument("--interpolation", default="linear", choices=INTERPOLATION_OPTIONS)
    parser.add_argument("--background-grid", type=int, default=20, help="Size of the sparse background sample lattice")
    parser.add_argument("--render-scale", type=int, default=12, help="Output resolution in samples per source cell")
    parser.add_argument(
        "--ambient-weight",
        type=float,
        default=0.5,
        help="Composite blend factor from 0.0 (all signal) to 1.0 (all ambient)",
    )
    parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian interpolation sigma")
    parser.add_argument("--idw-power", type=float, default=2.0, help="IDW power parameter")
    parser.add_argument("--cubic-a", type=float, default=-0.5, help="Parametric cubic kernel coefficient")
    parser.add_argument("--mask-invalid", action="store_true", help="Mask zones whose status is non-zero")
    parser.add_argument("--vmin", type=float, help="Manual color minimum")
    parser.add_argument("--vmax", type=float, help="Manual color maximum")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    viewer = CaptureJsonViewer()
    capture_path = Path(args.capture_json) if args.capture_json else None

    viewer.capture_path = Path(args.capture_file)
    viewer.capture_file_edit.setText(str(viewer.capture_path))
    viewer.serial_port_edit.setText(args.port)
    viewer.serial_baud_spin.setValue(args.baud)

    if capture_path is not None:
        viewer.load_capture(capture_path)

    viewer.dataset_combo.setCurrentText(args.dataset)
    viewer.overlay_combo.setCurrentText(args.overlay_readings)
    viewer.colormap_combo.setCurrentText(args.colormap)
    viewer.interpolation_combo.setCurrentText(args.interpolation)
    viewer.background_grid.setValue(args.background_grid)
    viewer.render_scale.setValue(args.render_scale)
    viewer.composite_ambient_weight.setValue(min(max(args.ambient_weight, 0.0), 1.0))
    viewer.blur_sigma.setValue(args.sigma)
    viewer.idw_power.setValue(args.idw_power)
    viewer.cubic_a.setValue(args.cubic_a)
    viewer.mask_invalid_checkbox.setChecked(args.mask_invalid)

    if args.vmin is not None and args.vmax is not None:
        viewer.auto_range_checkbox.setChecked(False)
        viewer.vmin_spin.setValue(args.vmin)
        viewer.vmax_spin.setValue(args.vmax)

    if args.export_png:
        if capture_path is None:
            raise SystemExit("capture_json is required with --export-png")
        viewer.refresh_view()
        viewer.canvas.save_png(Path(args.export_png))
        return 0

    if args.auto_connect:
        viewer.connect_serial()

    viewer.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
