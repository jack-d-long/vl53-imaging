# VL53L1X Max-Resolution ROI Sweep

This repository contains a SparkFun-library Arduino sketch that sweeps a `4x4` VL53L1X ROI across the verified `16x16` SPAD map and streams per-zone `signal`, `ambient`, `distance`, and `range status` as JSON. A Python visualizer renders live heatmaps for IR intensity and distance.

## Verified geometry

The SPAD numbering in the sketch is taken directly from ST UM2555 Rev 3, section 3.1. ST also states that when an ROI center falls between SPADs, you choose the SPAD "to the right, or above" the geometric center.

For a `4x4` ROI on a `16x16` array:

- Valid top-left ROI positions run from `(row=0..12, col=0..12)`.
- The resulting valid optical-center rows are `1..13`.
- The resulting valid optical-center columns are `2..14`.
- Dense step-1 sweep: `13 x 13 = 169` measurements.
- 50% overlap step-2 sweep: `7 x 7 = 49` measurements.
- Non-overlapping step-4 sweep: `4 x 4 = 16` measurements.

So the practical answer to the "is it 8x8?" question is: no, not for a verified `4x4` ROI. The regular 2-SPAD-stride grid is `7x7`, while the true maximum spatial sampling density is `13x13`.

## Files

- `vl53img.ino`: Arduino sketch for the SparkFun `SparkFun_VL53L1X` library.
- `serial_visualizer.py`: Live serial heatmap viewer using `pyserial`, `numpy`, and `matplotlib`.
- `capture_json_viewer.py`: Drag-and-drop offline viewer for saved frame JSON files using PyQt6.

## Arduino sketch behavior

- Uses the verified ST optical-center map as a hardcoded lookup table.
- Defaults to `ROI_WIDTH=4`, `ROI_HEIGHT=4`, `ROI_STEP_X=1`, `ROI_STEP_Y=1`.
- Sits idle until the host sends a serial command, then performs one full sweep and returns to idle.
- During a single requested frame, it uses continuous ranging and programs the next ROI immediately after each measurement so the next shot uses the new zone, following UM2555 timing guidance.
- Emits a startup `meta` JSON message describing the full sweep topology, then emits one `frame` JSON object per full sweep.
- Defaults to `SENSOR_MODE_INTENSITY`, which selects short distance mode with a `20 ms` timing budget.
- `SENSOR_MODE_DEPTH` switches to long distance mode with a `50 ms` timing budget.
- Supported serial commands are `capture`, `meta`, and `help`.

## Expected frame rate

Ignoring serial overhead, the frame time is approximately:

- Dense `13x13` sweep at `20 ms`: about `169 * 20 ms = 3380 ms`.
- `7x7` sweep at `20 ms`: about `49 * 20 ms = 980 ms`.
- `4x4` sweep at `20 ms`: about `16 * 20 ms = 320 ms`.

At `921600` baud the JSON transmission time is usually smaller than the ranging time, but dense mode is still a multi-second frame.

## Gotchas

- Per-zone calibration matters. UM2555 and AN5191 both note that offset calibration is ROI dependent; if you need better than a few centimeters, calibrate each zone separately.
- If you see frequent status `13` near edges, increase `INTER_MEASUREMENT_MS` slightly or back the sweep one SPAD inward.
- The SPAD map in the sketch matches ST's table orientation, not necessarily your final board orientation. If your assembled system rotates the sensor, rotate or flip in software.
- Dense `13x13` mode is best on ESP32 or STM32. It is possible to run on AVR, but SRAM headroom and serial throughput are much tighter.

## Python visualizer

Install dependencies:

```bash
python3 -m pip install pyserial numpy matplotlib
```

Run:

```bash
python3 serial_visualizer.py /dev/ttyUSB0
```

Change the port for your platform, for example `COM5` on Windows or `/dev/cu.usbserial-*` on macOS.

The viewer sends `meta` automatically on connect. Click the `Capture` button, or press `space` / `c`, to request one frame from the sensor.

## Offline JSON viewer

Use the offline viewer for saved capture files such as `latest_capture.json`:

```bash
python3 capture_json_viewer.py latest_capture.json
```

Or launch it with no arguments and drag a `.json` capture file onto the window.

Available controls:

- Dataset selector: `signal`, `ambient`, `distance`, `status`, `attempts`
- Render scale: expands the coarse sensor grid before display
- Auto or manual color range
- Colormap selection
- Gaussian blur sigma in sensor-cell units
- Interpolation mode
- Invalid-zone masking

## Optional extensions

- Super-resolution style reconstruction: use `ROI_STEP_X=1` and `ROI_STEP_Y=1`, then fuse overlapping measurements offline with interpolation or deconvolution.
- Servo raster scan: at each pan/tilt angle, collect one full ROI sweep and stitch each sub-grid into a larger image. This is a clean next step because the sketch already emits spatially indexed zones.

## References

- ST UM2555 Rev 3, "VL53L1X ultra lite driver multiple zone implementation": https://www.st.com/resource/en/user_manual/um2555-vl53l1x-ultra-lite-driver-multiple-zone-implementation-stmicroelectronics.pdf
- ST AN5191 Rev 1, "Using the programmable region of interest (ROI) with the VL53L1X": https://www.st.com/resource/en/application_note/an5191-using-the-programmable-region-of-interest-roi-with-the-vl53l1x-stmicroelectronics.pdf
- SparkFun VL53L1X Arduino library overview: https://learn.sparkfun.com/tutorials/qwiic-distance-sensor-vl53l1x-hookup-guide/arduino-library-overview
