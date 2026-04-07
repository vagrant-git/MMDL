from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent

REQUIRED_SENSOR_KEYS = (
    "serial_port",
    "sample_rate",
    "pressure_slope",
    "pressure_intercept",
    "flow_slope",
    "flow_intercept",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK3588 deployment preflight checks.")
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
        default=None,
        help="Calibration and serial config copied from the previous RK3588 project.",
    )
    parser.add_argument("--audio-device-index", type=int, default=None, help="PyAudio input device index to validate.")
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List available PyAudio input devices.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print_status("WARN", f"Ignoring runtime-only args during preflight: {' '.join(unknown)}")
    return args


def print_status(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def check_python_version() -> list[str]:
    failures: list[str] = []
    version_text = ".".join(str(part) for part in sys.version_info[:3])
    print_status("OK", f"python_version={version_text}")
    if sys.version_info < (3, 9):
        failures.append("Python 3.9 or newer is required.")
    return failures


def check_path(path_text: str, label: str) -> list[str]:
    failures: list[str] = []
    path = Path(path_text)
    if path.exists():
        print_status("OK", f"{label}={path}")
    else:
        failures.append(f"Missing {label}: {path}")
    return failures


def check_imports() -> tuple[list[str], dict[str, bool]]:
    failures: list[str] = []
    modules = {
        "numpy": "required",
        "yaml": "required",
        "onnxruntime": "required",
        "pyaudio": "required",
        "serial": "required",
        "torch": "optional",
        "torchaudio": "optional",
        "librosa": "optional",
        "scipy": "optional",
    }
    results: dict[str, bool] = {}
    for module_name, importance in modules.items():
        try:
            importlib.import_module(module_name)
            results[module_name] = True
            print_status("OK", f"import {module_name}")
        except Exception as exc:
            results[module_name] = False
            message = f"import {module_name} failed: {type(exc).__name__}: {exc}"
            if importance == "required":
                failures.append(message)
            else:
                print_status("WARN", message)

    if results.get("torch") and results.get("torchaudio"):
        print_status("OK", "torch/torchaudio preprocessing dependencies are installed")
    elif results.get("librosa") and results.get("scipy"):
        print_status("WARN", "torch path unavailable, runtime can fall back to librosa/scipy")
    else:
        failures.append("Neither torch/torchaudio nor librosa/scipy preprocessing dependencies are fully available.")
    return failures, results


def check_config(path_text: str) -> list[str]:
    failures: list[str] = []
    try:
        import yaml
    except Exception as exc:
        return [f"Cannot parse config without PyYAML: {type(exc).__name__}: {exc}"]

    path = Path(path_text)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Failed to read config {path}: {type(exc).__name__}: {exc}"]

    required_keys = ("audio_sample_rate", "sensor_sample_rate", "window_sec", "task")
    missing = [key for key in required_keys if key not in config]
    if missing:
        failures.append(f"Config missing keys: {', '.join(missing)}")
        return failures

    class_names = (config.get("task") or {}).get("class_names", [])
    print_status(
        "OK",
        "config_summary="
        f"audio_rate={config['audio_sample_rate']} "
        f"sensor_rate={config['sensor_sample_rate']} "
        f"window_sec={config['window_sec']} "
        f"class_names={class_names}",
    )
    return failures


def load_sensor_params(path_text: str | None) -> tuple[list[str], dict[str, object] | None]:
    if not path_text:
        return ["No --sensor-params supplied. Realtime inference cannot start without params.json."], None

    path = Path(path_text)
    if not path.exists():
        return [f"Missing sensor params: {path}"], None

    try:
        params = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Failed to parse sensor params {path}: {type(exc).__name__}: {exc}"], None

    missing = [key for key in REQUIRED_SENSOR_KEYS if key not in params]
    if missing:
        return [f"sensor params missing keys: {', '.join(missing)}"], None

    serial_port = Path(str(params["serial_port"]))
    if serial_port.is_absolute() and not serial_port.exists():
        return [f"serial_port does not exist on this machine: {serial_port}"], None

    print_status(
        "OK",
        f"sensor_params={path} serial_port={params['serial_port']} sample_rate={params['sample_rate']}",
    )
    return [], params


def check_sensor_params_against_config(config_path: str, sensor_params: dict[str, object] | None) -> list[str]:
    if sensor_params is None:
        return []
    try:
        import yaml
    except Exception:
        return []

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_rate = float(config["sensor_sample_rate"])
    source_rate = float(sensor_params["sample_rate"])
    if abs(model_rate - source_rate) > 1e-6:
        print_status(
            "WARN",
            f"sensor_sample_rate mismatch: params.json={source_rate}Hz config={model_rate}Hz; runtime will resample.",
        )
    return []


def list_audio_devices(selected_index: int | None) -> list[str]:
    failures: list[str] = []
    try:
        import pyaudio
    except Exception as exc:
        return [f"Cannot list audio devices without PyAudio: {type(exc).__name__}: {exc}"]

    pa = pyaudio.PyAudio()
    try:
        try:
            default_info = pa.get_default_input_device_info()
            print_status("OK", f"default_input_device={int(default_info['index'])}:{default_info['name']}")
        except OSError:
            print_status("WARN", "default input device unavailable")

        found = False
        selected_ok = selected_index is None
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            max_input_channels = int(info.get("maxInputChannels", 0))
            if max_input_channels <= 0:
                continue
            found = True
            print_status(
                "OK",
                f"audio_device index={idx} name={info.get('name', 'unknown')} "
                f"channels={max_input_channels} default_rate={int(info.get('defaultSampleRate', 0))}",
            )
            if selected_index == idx:
                selected_ok = True
        if not found:
            failures.append("No input-capable audio devices found.")
        if selected_index is not None and not selected_ok:
            failures.append(f"Requested audio device index {selected_index} was not found or is not input-capable.")
    finally:
        pa.terminate()
    return failures


def check_runtime_load(onnx_model: str, config_path: str) -> list[str]:
    try:
        from runtime_infer_onnx import RuntimeOnnxInfer
    except Exception as exc:
        return [f"Failed to import runtime_infer_onnx: {type(exc).__name__}: {exc}"]

    try:
        infer_engine = RuntimeOnnxInfer(onnx_model, config_path)
    except Exception as exc:
        return [f"Failed to initialize RuntimeOnnxInfer: {type(exc).__name__}: {exc}"]

    detail = infer_engine.preprocess_backend
    if infer_engine.preprocess_backend_reason:
        detail += f" ({infer_engine.preprocess_backend_reason})"
    print_status(
        "OK",
        "runtime_load="
        f"audio_samples={infer_engine.audio_samples} "
        f"sensor_samples={infer_engine.sensor_samples} "
        f"preprocess_backend={detail}",
    )
    return []


def main() -> int:
    args = parse_args()

    failures: list[str] = []
    failures.extend(check_python_version())
    failures.extend(check_path(args.onnx_model, "onnx_model"))
    failures.extend(check_path(args.config, "config"))

    import_failures, _ = check_imports()
    failures.extend(import_failures)
    failures.extend(check_config(args.config))
    if args.list_audio_devices and args.sensor_params is None:
        print_status("WARN", "Skipping sensor params validation while listing audio devices only.")
    else:
        sensor_failures, sensor_params = load_sensor_params(args.sensor_params)
        failures.extend(sensor_failures)
        failures.extend(check_sensor_params_against_config(args.config, sensor_params))

    if args.list_audio_devices or args.audio_device_index is not None:
        failures.extend(list_audio_devices(args.audio_device_index))

    if not failures:
        failures.extend(check_runtime_load(args.onnx_model, args.config))

    if failures:
        for message in failures:
            print_status("FAIL", message)
        return 1

    print_status("OK", "preflight checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
