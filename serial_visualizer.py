#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

if "MPLBACKEND" not in os.environ and sys.platform == "darwin":
    os.environ["MPLBACKEND"] = "TkAgg"

import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.widgets import Button


class LiveGridViewer:
    def __init__(self, ser: serial.Serial, capture_path: Path) -> None:
        self.ser = ser
        self.capture_path = capture_path
        self.rows: Optional[int] = None
        self.cols: Optional[int] = None
        self.signal_image = None
        self.distance_image = None
        self.signal_colorbar = None
        self.distance_colorbar = None
        self.title = None
        self.device_state = "idle"
        self.running = True

        plt.ion()
        self.figure, axes = plt.subplots(1, 2, figsize=(10, 5))
        self.figure.subplots_adjust(bottom=0.18)
        self.signal_axis, self.distance_axis = axes
        self.signal_axis.set_title("IR Signal (kcps)")
        self.distance_axis.set_title("Distance (mm)")
        self.signal_axis.set_xlabel("Grid column")
        self.signal_axis.set_ylabel("Grid row")
        self.distance_axis.set_xlabel("Grid column")
        self.distance_axis.set_ylabel("Grid row")
        self.capture_button_axis = self.figure.add_axes([0.24, 0.04, 0.16, 0.08])
        self.capture_button = Button(self.capture_button_axis, "Capture")
        self.capture_button.on_clicked(self.capture_requested)
        self.read_button_axis = self.figure.add_axes([0.42, 0.04, 0.16, 0.08])
        self.read_button = Button(self.read_button_axis, "Read")
        self.read_button.on_clicked(self.read_requested)
        self.exit_button_axis = self.figure.add_axes([0.60, 0.04, 0.16, 0.08])
        self.exit_button = Button(self.exit_button_axis, "Exit")
        self.exit_button.on_clicked(self.exit_requested)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key_press)
        plt.show(block=False)
        self.set_status("Waiting for device...")

    def set_status(self, text: str) -> None:
        label = f"{text}  |  state={self.device_state}"
        if self.title is None:
            self.title = self.figure.suptitle(label)
        else:
            self.title.set_text(label)
        self.figure.canvas.draw_idle()

    def service_ui(self) -> None:
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.01)

    def send_command(self, command: str) -> None:
        self.ser.write((command + "\n").encode("utf-8"))
        self.ser.flush()

    def capture_requested(self, _event=None) -> None:
        self.device_state = "requesting"
        self.set_status("Capture requested")
        self.send_command("capture")

    def read_requested(self, _event=None) -> None:
        if not self.capture_path.exists():
            self.set_status(f"No capture file: {self.capture_path.name}")
            return

        try:
            payload = json.loads(self.capture_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.set_status(f"Failed to read {self.capture_path.name}")
            return

        if payload.get("type") != "frame":
            self.set_status(f"Invalid capture file: {self.capture_path.name}")
            return

        if getattr(self, "print_zones_on_read", False):
            print_zone_dump(payload)
        self.handle_frame(payload)

    def exit_requested(self, _event=None) -> None:
        self.running = False
        plt.close(self.figure)

    def on_key_press(self, event) -> None:
        if event.key in (" ", "c"):
            self.capture_requested()
        elif event.key == "r":
            self.read_requested()
        elif event.key in ("q", "escape"):
            self.exit_requested()

    def handle_meta(self, payload: dict) -> None:
        rows = payload.get("rows")
        cols = payload.get("cols")
        if isinstance(rows, int) and isinstance(cols, int):
            self.rows = rows
            self.cols = cols
        self.set_status("Metadata received")

    def handle_state(self, payload: dict) -> None:
        state = payload.get("state")
        if isinstance(state, str):
            self.device_state = state
            if state == "idle":
                self.set_status("Ready")
            elif state == "capturing":
                self.set_status("Capturing")
            else:
                self.set_status("Device update")

    def handle_frame(self, payload: dict) -> None:
        rows = payload["rows"]
        cols = payload["cols"]
        zones = payload["zones"]
        frame_id = payload["frame"]
        frame_time_ms = payload["frame_time_ms"]

        signal = np.full((rows, cols), np.nan, dtype=float)
        distance = np.full((rows, cols), np.nan, dtype=float)

        for zone in zones:
            row = zone["grid_row"]
            col = zone["grid_col"]
            status = zone["status"]
            if status == 0:
                signal[row, col] = zone["signal_kcps"]
                distance[row, col] = zone["distance_mm"]

        self._update_plot(signal, distance, frame_id, frame_time_ms)

    def _update_plot(self, signal: np.ndarray, distance: np.ndarray, frame_id: int, frame_time_ms: float) -> None:
        if self.signal_image is None:
            self.signal_image = self.signal_axis.imshow(signal, cmap="inferno", interpolation="nearest")
            self.distance_image = self.distance_axis.imshow(distance, cmap="viridis_r", interpolation="nearest")
            self.signal_colorbar = self.figure.colorbar(self.signal_image, ax=self.signal_axis, fraction=0.046, pad=0.04)
            self.distance_colorbar = self.figure.colorbar(self.distance_image, ax=self.distance_axis, fraction=0.046, pad=0.04)
            self.signal_colorbar.set_label("kcps")
            self.distance_colorbar.set_label("mm")
        else:
            self.signal_image.set_data(signal)
            self.distance_image.set_data(distance)

        finite_signal = signal[np.isfinite(signal)]
        finite_distance = distance[np.isfinite(distance)]

        if finite_signal.size:
            self.signal_image.set_clim(vmin=float(np.nanmin(finite_signal)), vmax=float(np.nanpercentile(finite_signal, 98)))
        if finite_distance.size:
            self.distance_image.set_clim(vmin=float(np.nanmin(finite_distance)), vmax=float(np.nanpercentile(finite_distance, 98)))

        self.device_state = "idle"
        self.set_status(f"Frame {frame_id}  |  {frame_time_ms:.0f} ms")
        self.service_ui()

    def save_frame(self, payload: dict) -> None:
        self.capture_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.set_status(f"Saved {self.capture_path.name}")


def open_serial(port: str, baud: int) -> serial.Serial:
    try:
        return serial.Serial(port, baudrate=baud, timeout=0.05)
    except serial.SerialException as exc:
        raise SystemExit(f"Failed to open {port}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live VL53L1X ROI-sweep visualizer")
    parser.add_argument("port", help="Serial port, for example /dev/ttyUSB0, /dev/cu.usbserial-*, or COM5")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--echo-json", action="store_true", help="Print each parsed JSON message to stdout")
    parser.add_argument("--log-jsonl", help="Append each parsed JSON message to a JSONL file")
    parser.add_argument("--print-zones", action="store_true", help="Print per-zone parsed values for each frame")
    parser.add_argument("--capture-file", default="latest_capture.json", help="Path for saved frame JSON")
    return parser.parse_args()


def print_zone_dump(payload: dict) -> None:
    frame_id = payload["frame"]
    for zone in payload["zones"]:
        print(
            "frame={frame} row={row} col={col} spad={spad} signal_kcps={signal} ambient_kcps={ambient} distance_mm={distance} status={status}".format(
                frame=frame_id,
                row=zone["grid_row"],
                col=zone["grid_col"],
                spad=zone["spad_center"],
                signal=zone["signal_kcps"],
                ambient=zone["ambient_kcps"],
                distance=zone["distance_mm"],
                status=zone["status"],
            )
        )


def main() -> int:
    args = parse_args()
    log_file = open(args.log_jsonl, "a", encoding="utf-8") if args.log_jsonl else None
    capture_path = Path(args.capture_file)

    try:
        with open_serial(args.port, args.baud) as ser:
            viewer = LiveGridViewer(ser, capture_path)
            viewer.print_zones_on_read = args.print_zones
            time.sleep(1.0)
            ser.reset_input_buffer()
            viewer.send_command("meta")
            buffer = bytearray()

            while viewer.running and plt.fignum_exists(viewer.figure.number):
                raw = ser.read(ser.in_waiting or 1)
                if not raw:
                    viewer.service_ui()
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
                        print(f"Skipping non-JSON line: {line}", file=sys.stderr)
                        continue

                    if args.echo_json:
                        print(json.dumps(payload, separators=(",", ":")))

                    if log_file is not None:
                        log_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
                        log_file.flush()

                    message_type = payload.get("type")
                    if message_type == "meta":
                        viewer.handle_meta(payload)
                    elif message_type == "state":
                        viewer.handle_state(payload)
                    elif message_type == "frame":
                        viewer.save_frame(payload)
                    else:
                        viewer.service_ui()
    finally:
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
