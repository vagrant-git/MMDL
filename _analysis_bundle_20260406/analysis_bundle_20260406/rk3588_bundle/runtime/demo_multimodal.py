from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from math import gcd
from pathlib import Path

import numpy as np
import pyaudio

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

    def latest_available(self, length: int) -> np.ndarray:
        available = self.max_samples if self.full else self.write_pos
        if available <= 0:
            return np.zeros(0, dtype=np.int16)
        return self.latest(min(length, available))


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


def resample_audio_int16(audio: np.ndarray, input_rate: int, target_rate: int, target_length: int) -> np.ndarray:
    if input_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape={audio.shape}")
    if audio.shape[0] < 2:
        raise ValueError(f"Need at least 2 audio samples to resample, got {audio.shape[0]}")
    if input_rate == target_rate and audio.shape[0] == target_length:
        return audio.astype(np.int16, copy=False)

    src_pos = np.linspace(0.0, 1.0, num=audio.shape[0], dtype=np.float32)
    dst_pos = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    resampled = np.interp(dst_pos, src_pos, audio.astype(np.float32, copy=False))
    return np.clip(np.round(resampled), -32768, 32767).astype(np.int16)


def normalization_stats_from_values(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 1.0}
    values = values.astype(np.float32, copy=False)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-6:
        std = 1.0
    return {"mean": mean, "std": std}


def print_input_devices(pa: pyaudio.PyAudio) -> None:
    try:
        default_info = pa.get_default_input_device_info()
        print(
            f"default_input_device_index={int(default_info['index'])} "
            f"name={default_info['name']}",
            flush=True,
        )
    except OSError:
        print("default_input_device_index=unavailable", flush=True)

    found = False
    for idx in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(idx)
        max_input_channels = int(info.get("maxInputChannels", 0))
        if max_input_channels <= 0:
            continue
        found = True
        print(
            f"index={idx} name={info.get('name', 'unknown')} "
            f"channels={max_input_channels} default_rate={int(info.get('defaultSampleRate', 0))}",
            flush=True,
        )
    if not found:
        print("no input-capable audio devices found", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK3588 realtime multimodal ONNX demo.")
    parser.add_argument("--audio-rate", type=int, default=16000)
    parser.add_argument(
        "--audio-input-rate",
        type=int,
        default=None,
        help="Physical input device sample rate. Defaults to --audio-rate unless a selected device reports a different default rate.",
    )
    parser.add_argument("--audio-chunk", type=int, default=1024)
    parser.add_argument("--audio-device-index", type=int, default=None, help="PyAudio input device index.")
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
        default=str(BUNDLE_ROOT / "R_Identification" / "params.json"),
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
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List available PyAudio input devices and exit.",
    )
    parser.add_argument(
        "--normalization-mode",
        choices=("window", "rolling"),
        default="rolling",
        help="Use per-window normalization or rolling long-buffer statistics before inference.",
    )
    parser.add_argument(
        "--normalization-buffer-sec",
        type=float,
        default=30.0,
        help="Context duration used when --normalization-mode=rolling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pa = pyaudio.PyAudio()
    if args.list_audio_devices:
        try:
            print_input_devices(pa)
        finally:
            pa.terminate()
        return

    from runtime_infer_onnx import RuntimeOnnxInfer
    from sensor_serial import SerialPQReader, SensorRingBuffer

    output_dir = Path(args.output_dir)
    if args.save_snapshots:
        output_dir.mkdir(parents=True, exist_ok=True)

    infer_engine = RuntimeOnnxInfer(args.onnx_model, args.config)
    if args.audio_rate != infer_engine.audio_rate:
        raise ValueError(f"--audio-rate={args.audio_rate} does not match config audio rate {infer_engine.audio_rate}")
    if abs(args.window_sec - infer_engine.window_sec) > 1e-6:
        raise ValueError(f"--window-sec={args.window_sec} does not match config window {infer_engine.window_sec}")
    normalization_buffer_sec = max(args.window_sec, args.normalization_buffer_sec)
    print(f"audio_preprocess_backend={infer_engine.preprocess_backend}", flush=True)
    print(
        "audio_preprocess_backend_mode="
        f"{'training_consistent_torch' if infer_engine.preprocess_backend == 'torch' else 'librosa_fallback'}",
        flush=True,
    )
    if infer_engine.preprocess_backend_reason:
        print(f"audio_preprocess_backend_detail={infer_engine.preprocess_backend_reason}", flush=True)
    print(f"input_normalization_mode={args.normalization_mode}", flush=True)
    if args.normalization_mode == "rolling":
        print(f"input_normalization_buffer_sec={normalization_buffer_sec:.1f}", flush=True)

    audio_samples = infer_engine.audio_samples
    sensor_reader = SerialPQReader(args.sensor_params)
    sensor_reader.open()
    sensor_window_samples = max(2, int(round(sensor_reader.sample_rate * args.window_sec)))
    sensor_norm_samples = max(2, int(round(sensor_reader.sample_rate * normalization_buffer_sec)))
    if sensor_window_samples != infer_engine.sensor_samples:
        print(
            "sensor_window_resample="
            f"source_rate={sensor_reader.sample_rate}Hz "
            f"source_samples={sensor_window_samples} "
            f"model_rate={infer_engine.sensor_rate}Hz "
            f"model_samples={infer_engine.sensor_samples}",
            flush=True,
        )
    sensor_buffer = SensorRingBuffer(
        max_seconds=max(args.window_sec * 2.0, normalization_buffer_sec, 10.0),
        sample_rate=sensor_reader.sample_rate,
    )

    stream = None
    input_audio_rate = args.audio_rate if args.audio_input_rate is None else args.audio_input_rate
    if args.audio_device_index is not None:
        info = pa.get_device_info_by_index(args.audio_device_index)
        if int(info.get("maxInputChannels", 0)) <= 0:
            raise ValueError(f"Audio device index {args.audio_device_index} is not input-capable.")
        if args.audio_input_rate is None:
            input_audio_rate = int(round(float(info.get("defaultSampleRate", args.audio_rate))))
        print(f"using audio device index={args.audio_device_index} name={info.get('name', 'unknown')}", flush=True)
    capture_audio_samples = max(2, int(round(input_audio_rate * args.window_sec)))
    capture_audio_norm_samples = max(2, int(round(input_audio_rate * normalization_buffer_sec)))
    capture_audio_chunk = max(1, int(round(args.audio_chunk * input_audio_rate / args.audio_rate)))
    if input_audio_rate != args.audio_rate:
        print(
            "audio_input_resample="
            f"source_rate={input_audio_rate}Hz "
            f"source_samples={capture_audio_samples} "
            f"model_rate={args.audio_rate}Hz "
            f"model_samples={audio_samples}",
            flush=True,
        )
    audio_buffer = AudioRingBuffer(
        max_samples=max(capture_audio_samples * 2, capture_audio_norm_samples + capture_audio_chunk)
    )

    open_kwargs = {
        "format": pyaudio.paInt16,
        "channels": 1,
        "rate": input_audio_rate,
        "input": True,
        "frames_per_buffer": capture_audio_chunk,
    }
    if args.audio_device_index is not None:
        open_kwargs["input_device_index"] = args.audio_device_index
    stream = pa.open(**open_kwargs)

    next_infer_ts = time.perf_counter() + args.window_sec
    print("realtime multimodal ONNX demo started", flush=True)
    try:
        while True:
            try:
                raw = stream.read(capture_audio_chunk, exception_on_overflow=False)
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

            target_sensor_len = sensor_window_samples
            if not sensor_buffer.ready(target_sensor_len):
                print("sensor buffer not ready", flush=True)
                continue

            audio_window = audio_buffer.latest(capture_audio_samples)
            if input_audio_rate != args.audio_rate:
                audio_window = resample_audio_int16(audio_window, input_audio_rate, args.audio_rate, audio_samples)
            pressure = np.asarray(list(sensor_buffer.pressure)[-target_sensor_len:], dtype=np.float32)
            flow = np.asarray(list(sensor_buffer.flow)[-target_sensor_len:], dtype=np.float32)
            normalization_stats = None
            if args.normalization_mode == "rolling":
                audio_context = audio_buffer.latest_available(capture_audio_norm_samples)
                if audio_context.shape[0] >= 2 and input_audio_rate != args.audio_rate:
                    target_audio_context_len = max(2, int(round(audio_context.shape[0] * args.audio_rate / input_audio_rate)))
                    audio_context = resample_audio_int16(
                        audio_context,
                        input_audio_rate,
                        args.audio_rate,
                        target_audio_context_len,
                    )
                pressure_context = np.asarray(list(sensor_buffer.pressure)[-sensor_norm_samples:], dtype=np.float32)
                flow_context = np.asarray(list(sensor_buffer.flow)[-sensor_norm_samples:], dtype=np.float32)
                normalization_stats = {
                    "audio": normalization_stats_from_values(
                        audio_context.astype(np.float32, copy=False) / 32768.0
                    ),
                    "pressure": normalization_stats_from_values(pressure_context),
                    "flow": normalization_stats_from_values(flow_context),
                }
            infer_start = time.perf_counter()
            result = infer_engine.infer(audio_window, pressure, flow, normalization_stats=normalization_stats)
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
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()
        sensor_reader.close()


if __name__ == "__main__":
    main()
