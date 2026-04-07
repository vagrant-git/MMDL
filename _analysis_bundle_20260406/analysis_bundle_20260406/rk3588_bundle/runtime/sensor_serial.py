from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None


@dataclass
class SensorSample:
    timestamp_sec: float
    pressure: float
    flow: float


def parse_serial_line(line: str) -> tuple[str | None, float | None]:
    try:
        channel, value = line.strip().split(":", maxsplit=1)
        value = value.strip().replace("V", "").replace("v", "")
        if channel not in {"CH0", "CH1"}:
            return None, None
        return channel, float(value)
    except Exception:
        return None, None


class SerialPQReader:
    def __init__(self, params_json: str | Path, baudrate: int = 115200) -> None:
        self.params_path = Path(params_json)
        params = json.loads(self.params_path.read_text(encoding="utf-8"))
        self.serial_port = str(params["serial_port"])
        self.sample_rate = float(params["sample_rate"])
        self.pressure_slope = float(params["pressure_slope"])
        self.pressure_intercept = float(params["pressure_intercept"])
        self.flow_slope = float(params["flow_slope"])
        self.flow_intercept = float(params["flow_intercept"])
        self.baudrate = baudrate
        self._ser = None
        self._current_measurement: dict[str, float] = {}
        self._line_buffer = ""

    def open(self) -> None:
        if serial is None:
            raise RuntimeError("pyserial is required for serial PQ capture.")
        self._ser = serial.Serial(self.serial_port, self.baudrate, timeout=1)

    def close(self) -> None:
        if self._ser is not None and self._ser.is_open:
            self._ser.close()

    def _build_sample(self, channel: str | None, voltage: float | None) -> SensorSample | None:
        if channel is None or voltage is None:
            return None
        if channel == "CH0":
            self._current_measurement["pressure_voltage"] = voltage
        elif channel == "CH1":
            self._current_measurement["flow_voltage"] = voltage
        if "pressure_voltage" not in self._current_measurement or "flow_voltage" not in self._current_measurement:
            return None
        pressure = self.pressure_slope * self._current_measurement["pressure_voltage"] + self.pressure_intercept
        flow = self.flow_slope * self._current_measurement["flow_voltage"] + self.flow_intercept
        self._current_measurement.clear()
        return SensorSample(timestamp_sec=time.perf_counter(), pressure=pressure, flow=flow)

    def read_sample(self) -> SensorSample | None:
        if self._ser is None:
            raise RuntimeError("Serial port is not open.")
        line = self._ser.readline().decode("utf-8", errors="ignore")
        if not line:
            return None
        channel, voltage = parse_serial_line(line)
        return self._build_sample(channel, voltage)

    def drain_samples(self) -> list[SensorSample]:
        if self._ser is None:
            raise RuntimeError("Serial port is not open.")
        waiting = int(self._ser.in_waiting)
        if waiting <= 0:
            return []
        chunk = self._ser.read(waiting).decode("utf-8", errors="ignore")
        if not chunk:
            return []
        self._line_buffer += chunk
        lines = self._line_buffer.splitlines(keepends=True)
        complete_lines: list[str] = []
        remainder = ""
        for line in lines:
            if line.endswith("\n") or line.endswith("\r"):
                complete_lines.append(line)
            else:
                remainder += line
        self._line_buffer = remainder

        samples: list[SensorSample] = []
        for line in complete_lines:
            channel, voltage = parse_serial_line(line)
            sample = self._build_sample(channel, voltage)
            if sample is not None:
                samples.append(sample)
        return samples


class SensorRingBuffer:
    def __init__(self, max_seconds: float, sample_rate: float) -> None:
        capacity = max(1, int(max_seconds * sample_rate) + 4)
        self.pressure = deque(maxlen=capacity)
        self.flow = deque(maxlen=capacity)
        self.time_sec = deque(maxlen=capacity)

    def append(self, sample: SensorSample) -> None:
        self.time_sec.append(sample.timestamp_sec)
        self.pressure.append(sample.pressure)
        self.flow.append(sample.flow)

    def ready(self, target_length: int) -> bool:
        return len(self.pressure) >= target_length and len(self.flow) >= target_length
