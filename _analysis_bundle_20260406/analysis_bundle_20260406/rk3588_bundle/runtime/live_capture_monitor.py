from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyaudio

from sensor_serial import SerialPQReader


DEFAULT_SENSOR_PARAMS = Path(__file__).resolve().parents[1] / "R_Identification" / "params.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live terminal monitor for PQ and audio capture.")
    parser.add_argument(
        "--sensor-params",
        default=str(DEFAULT_SENSOR_PARAMS),
        help="Path to params.json",
    )
    parser.add_argument("--audio-device-index", type=int, default=None, help="PyAudio input device index")
    parser.add_argument("--audio-input-rate", type=int, default=None, help="Physical audio capture rate")
    parser.add_argument("--audio-chunk", type=int, default=1024)
    parser.add_argument("--refresh-sec", type=float, default=0.5)
    return parser.parse_args()


def print_capture_config(device_name: str, input_audio_rate: int, sensor_reader: SerialPQReader) -> None:
    print(
        f"audio_device={device_name} input_rate={input_audio_rate}Hz "
        f"serial_port={sensor_reader.serial_port} sensor_rate={sensor_reader.sample_rate}Hz",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    sensor_reader = SerialPQReader(args.sensor_params)
    sensor_reader.open()

    pa = pyaudio.PyAudio()
    input_audio_rate = args.audio_input_rate
    device_name = "default"
    if args.audio_device_index is not None:
        info = pa.get_device_info_by_index(args.audio_device_index)
        if int(info.get("maxInputChannels", 0)) <= 0:
            raise ValueError(f"Audio device index {args.audio_device_index} is not input-capable.")
        device_name = str(info.get("name", "unknown"))
        if input_audio_rate is None:
            input_audio_rate = int(round(float(info.get("defaultSampleRate", 44100))))
    if input_audio_rate is None:
        input_audio_rate = 16000

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=input_audio_rate,
        input=True,
        frames_per_buffer=args.audio_chunk,
        input_device_index=args.audio_device_index,
    )

    latest_pressure = None
    latest_flow = None
    audio_start_ts = None
    pq_start_ts = None
    next_refresh = time.perf_counter()
    next_config_print = next_refresh + 10.0

    print("Board Live Monitor", flush=True)
    print_capture_config(device_name, input_audio_rate, sensor_reader)
    print("Streaming audio_sec / pq_sec / pressure / flow. Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            stream.read(args.audio_chunk, exception_on_overflow=False)
            if audio_start_ts is None:
                audio_start_ts = time.perf_counter()

            for sample in sensor_reader.drain_samples():
                if pq_start_ts is None:
                    pq_start_ts = sample.timestamp_sec
                latest_pressure = float(sample.pressure)
                latest_flow = float(sample.flow)

            now = time.perf_counter()
            if now >= next_config_print:
                print_capture_config(device_name, input_audio_rate, sensor_reader)
                next_config_print = now + 10.0
            if now < next_refresh:
                continue
            next_refresh = now + args.refresh_sec

            audio_sec = 0.0 if audio_start_ts is None else max(0.0, now - audio_start_ts)
            pq_sec = 0.0 if pq_start_ts is None else max(0.0, now - pq_start_ts)
            pressure_text = f"{latest_pressure:.3f} cmH2O" if latest_pressure is not None else "waiting..."
            flow_text = f"{latest_flow:.3f} L/min" if latest_flow is not None else "waiting..."
            print(
                f"audio_sec={audio_sec:.2f} pq_sec={pq_sec:.2f} "
                f"pressure={pressure_text} flow={flow_text}",
                flush=True,
            )
    except KeyboardInterrupt:
        return 0
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        sensor_reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
