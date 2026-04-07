from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pyaudio

from runtime_infer_onnx import RuntimeOnnxInfer
from sensor_serial import SerialPQReader, SensorRingBuffer

SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent


class AudioRingBuffer:
    def __init__(self, max_samples: int) -> None:
        self.buffer = np.zeros(max_samples, dtype=np.int16)
        self.max_samples = max_samples
        self.write_pos = 0
        self.full = False

    def append(self, chunk: np.ndarray) -> None:
        for value in chunk:
            self.buffer[self.write_pos] = value
            self.write_pos = (self.write_pos + 1) % self.max_samples
            if self.write_pos == 0:
                self.full = True

    def latest(self, length: int) -> np.ndarray:
        if length > self.max_samples:
            raise ValueError("Requested length exceeds buffer size.")
        if not self.full and self.write_pos < length:
            padded = np.zeros(length, dtype=np.int16)
            padded[-self.write_pos :] = self.buffer[: self.write_pos]
            return padded
        start = (self.write_pos - length) % self.max_samples
        if start < self.write_pos:
            return self.buffer[start : self.write_pos].copy()
        return np.concatenate([self.buffer[start:], self.buffer[: self.write_pos]]).astype(np.int16, copy=False)


def format_result(result: dict[str, object]) -> str:
    probs = result["probabilities"]
    if isinstance(probs, list):
        prob_text = ", ".join(f"{float(x):.4f}" for x in probs)
    else:
        prob_text = str(probs)
    return (
        f"pred={result['predicted_label']} "
        f"(idx={result['predicted_index']}) "
        f"probs=[{prob_text}]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK3588 realtime multimodal ONNX demo.")
    parser.add_argument("--audio-rate", type=int, default=16000)
    parser.add_argument("--audio-chunk", type=int, default=1024)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument(
        "--onnx-model",
        default=str(BUNDLE_ROOT / "models" / "hcaf_pcen_dualxattn.onnx"),
        help="Exported ONNX model path.",
    )
    parser.add_argument(
        "--config",
        default=str(BUNDLE_ROOT / "configs" / "final_model_unified_evidence.yaml"),
        help="Training-time config used to define preprocessing and labels.",
    )
    parser.add_argument(
        "--sensor-params",
        default="R_Identification/params.json",
        help="Calibration and serial config copied from the previous RK3588 project.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BUNDLE_ROOT / "runtime_debug"),
        help="Where to dump temporary audio/pq snapshots for the offline infer engine.",
    )
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Save each runtime window as wav/csv/json for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.save_snapshots:
        output_dir.mkdir(parents=True, exist_ok=True)

    infer_engine = RuntimeOnnxInfer(args.onnx_model, args.config)
    if args.audio_rate != infer_engine.audio_rate:
        raise ValueError(f"--audio-rate={args.audio_rate} does not match config audio rate {infer_engine.audio_rate}")
    if abs(args.window_sec - infer_engine.window_sec) > 1e-6:
        raise ValueError(f"--window-sec={args.window_sec} does not match config window {infer_engine.window_sec}")

    audio_samples = infer_engine.audio_samples
    sensor_reader = SerialPQReader(args.sensor_params)
    sensor_reader.open()
    sensor_buffer = SensorRingBuffer(max_seconds=max(args.window_sec * 2.0, 10.0), sample_rate=sensor_reader.sample_rate)
    audio_buffer = AudioRingBuffer(max_samples=max(audio_samples * 2, audio_samples + args.audio_chunk))

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=args.audio_rate,
        input=True,
        frames_per_buffer=args.audio_chunk,
    )

    next_infer_ts = time.perf_counter() + args.window_sec
    print("realtime multimodal ONNX demo started", flush=True)
    try:
        while True:
            try:
                raw = stream.read(args.audio_chunk, exception_on_overflow=False)
            except OSError as exc:
                print(f"audio read failed: {exc}", file=sys.stderr, flush=True)
                continue
            audio_buffer.append(np.frombuffer(raw, dtype=np.int16))

            while True:
                sample = sensor_reader.read_sample()
                if sample is None:
                    break
                sensor_buffer.append(sample)

            now = time.perf_counter()
            if now < next_infer_ts:
                continue
            next_infer_ts = now + args.hop_sec

            target_sensor_len = infer_engine.sensor_samples
            if not sensor_buffer.ready(target_sensor_len):
                print("sensor buffer not ready", flush=True)
                continue

            audio_window = audio_buffer.latest(audio_samples)
            pressure = np.asarray(list(sensor_buffer.pressure)[-target_sensor_len:], dtype=np.float32)
            flow = np.asarray(list(sensor_buffer.flow)[-target_sensor_len:], dtype=np.float32)
            infer_start = time.perf_counter()
            result = infer_engine.infer(audio_window, pressure, flow)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            print(f"{format_result(result)} latency_ms={infer_ms:.2f}", flush=True)

            if args.save_snapshots:
                rel_time = np.arange(target_sensor_len, dtype=np.float32) / float(sensor_reader.sample_rate)
                ts = int(time.time())
                wav_path = output_dir / f"{ts}_audio.wav"
                with wave.open(str(wav_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(args.audio_rate)
                    wf.writeframes(audio_window.tobytes())

                csv_path = output_dir / f"{ts}_daq.csv"
                csv_path.write_text(
                    "Time (s),Pressure (cmH2O),Flowrate (L/min)\n"
                    + "\n".join(
                        f"{t:.6f},{p:.6f},{f:.6f}" for t, p, f in zip(rel_time.tolist(), pressure.tolist(), flow.tolist())
                    ),
                    encoding="utf-8",
                )
                meta_path = output_dir / f"{ts}_meta.json"
                meta_path.write_text(
                    json.dumps(
                        {
                            "audio_rate": args.audio_rate,
                            "sensor_rate": sensor_reader.sample_rate,
                            "window_sec": args.window_sec,
                            "predicted_index": result["predicted_index"],
                            "predicted_label": result["predicted_label"],
                            "probabilities": result["probabilities"],
                            "wav_path": str(wav_path),
                            "daq_csv_path": str(csv_path),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"saved snapshot: {wav_path.name}, {csv_path.name}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        sensor_reader.close()


if __name__ == "__main__":
    main()
