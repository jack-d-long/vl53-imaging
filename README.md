# Photos from the VL53L1X Distance Sensor

This repository contains a SparkFun-library Arduino sketch that sweeps a 4x4 region of interest (ROI) across the 16x16 single photon avalanche diode (SPAD) array of the VL53L1X and streams per-zone `signal`, `ambient`, `distance`, and `range status` as JSON. A Python visualizer renders images for IR intensity and distance.

'signal' is better for objects primarily illuminated by the sensor itself. Photographs of landscapes or other ambiently-lit environments are better constructed with 'ambient'. 

## The Sensor

The SPAD numbering in the sketch is taken directly from ST UM2555 Rev 3, section 3.1. ST also states that when an ROI center falls between SPADs, you choose the SPAD "to the right, or above" the geometric center.

For a `4x4` ROI on a `16x16` array:

- Valid top-left ROI positions run from `(row=0..12, col=0..12)`.
- The resulting valid optical-center rows are `1..13`.
- The resulting valid optical-center columns are `2..14`.
- Dense step-1 sweep: `13 x 13 = 169` measurements.
- 50% overlap step-2 sweep: `7 x 7 = 49` measurements.
- Non-overlapping step-4 sweep: `4 x 4 = 16` measurements.

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

The same viewer can also talk directly to the device and save fresh captures through the existing JSON rendering pipeline:

```bash
python3 capture_json_viewer.py --port /dev/ttyUSB0 --auto-connect
```

The current Arduino sketch in this repo is configured for `115200` baud, so the viewer defaults to that rate.

Available controls:

- Serial port / baud / capture file: connection settings for live capture
- Connect / Disconnect / Capture / Read Capture File: device control without switching to `serial_visualizer.py`
- Dataset selector: `signal`, `ambient`, `composite`, `distance`, `status`, `attempts`
- Ambient weight: for the `composite` dataset, `0.0` is all signal and `1.0` is all ambient
- Reading overlay: `none`, `signal`, `ambient`, `signal + ambient`
- Colormap presets: `inferno`, `afmhot`, `gist_heat`, `viridis`, `magma`, `gray`
- Background lattice: size of the sparse sample grid filled with background samples
- Samples per cell: output resolution of the reconstructed field
- Interpolation method: `linear`, `nearest`, `cubic`, `idw`, `gaussian`
- Method-specific controls appear only for the selected interpolation
- Auto or manual color range
- Editable colormap selection
- Gaussian sigma for the `gaussian` interpolator
- IDW power for the `idw` interpolator
- Cubic `a` parameter for the `cubic` interpolator
- Invalid-zone masking
- Matplotlib toolbar zoom/pan, with `Save PNG` preserving the current zoomed view

You can also include overlays when exporting directly:

```bash
python3 capture_json_viewer.py latest_capture.json --dataset "Composite (Signal + Ambient)" --ambient-weight 0.65 --overlay-readings "Signal + Ambient" --export-png latest_capture.png
```

## Future work 

- use `ROI_STEP_X=1` and `ROI_STEP_Y=1`, then fuse overlapping measurements offline with interpolation or deconvolution.


## References

- ST UM2555 Rev 3, "VL53L1X ultra lite driver multiple zone implementation": https://www.st.com/resource/en/user_manual/um2555-vl53l1x-ultra-lite-driver-multiple-zone-implementation-stmicroelectronics.pdf
- ST AN5191 Rev 1, "Using the programmable region of interest (ROI) with the VL53L1X": https://www.st.com/resource/en/application_note/an5191-using-the-programmable-region-of-interest-roi-with-the-vl53l1x-stmicroelectronics.pdf
- SparkFun VL53L1X Arduino library overview: https://learn.sparkfun.com/tutorials/qwiic-distance-sensor-vl53l1x-hookup-guide/arduino-library-overview
