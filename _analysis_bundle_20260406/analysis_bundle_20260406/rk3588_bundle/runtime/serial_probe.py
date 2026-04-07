from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sensor_serial import parse_serial_line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe CH0/CH1 serial data without running the model.")
    parser.add_argument("--sensor-params", required=True, help="Path to R_Identification/params.json")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--duration-sec", type=float, default=10.0, help="How long to listen before exiting.")
    parser.add_argument("--max-pairs", type=int, default=20, help="Stop after this many pressure/flow pairs.")
    parser.add_argument("--quiet-raw", action="store_true", help="Do not print every raw serial line.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import serial
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyserial is required for serial probing.") from exc

    params_path = Path(args.sensor_params)
    params = json.loads(params_path.read_text(encoding="utf-8"))
    serial_port = str(params["serial_port"])
    pressure_slope = float(params["pressure_slope"])
    pressure_intercept = float(params["pressure_intercept"])
    flow_slope = float(params["flow_slope"])
    flow_intercept = float(params["flow_intercept"])

    ser = serial.Serial(serial_port, args.baudrate, timeout=1)
    current_measurement: dict[str, float] = {}
    pair_count = 0
    deadline = time.monotonic() + args.duration_sec

    print(
        f"serial probe started port={serial_port} baudrate={args.baudrate} "
        f"duration_sec={args.duration_sec} max_pairs={args.max_pairs}",
        flush=True,
    )
    try:
        while time.monotonic() < deadline and pair_count < args.max_pairs:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                continue
            if not args.quiet_raw:
                print(f"raw={raw_line}", flush=True)

            channel, voltage = parse_serial_line(raw_line)
            if channel is None or voltage is None:
                continue
            if channel == "CH0":
                current_measurement["pressure_voltage"] = voltage
            elif channel == "CH1":
                current_measurement["flow_voltage"] = voltage

            if "pressure_voltage" not in current_measurement or "flow_voltage" not in current_measurement:
                continue

            pressure = pressure_slope * current_measurement["pressure_voltage"] + pressure_intercept
            flow = flow_slope * current_measurement["flow_voltage"] + flow_intercept
            pair_count += 1
            print(
                f"pair={pair_count} "
                f"pressure_voltage={current_measurement['pressure_voltage']:.6f} "
                f"flow_voltage={current_measurement['flow_voltage']:.6f} "
                f"pressure={pressure:.6f} "
                f"flow={flow:.6f}",
                flush=True,
            )
            current_measurement.clear()
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

    print(f"serial probe finished pairs={pair_count}", flush=True)
    return 0 if pair_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
