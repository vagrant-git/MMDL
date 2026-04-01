from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import numpy as np


def load_flow(daq_path: Path) -> tuple[np.ndarray, np.ndarray]:
    times = []
    flows = []
    with daq_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["Time (s)"]))
            flows.append(float(row["Flowrate (L/min)"]))
    time_array = np.asarray(times, dtype=np.float32)
    time_array = time_array - time_array[0]
    return time_array, np.asarray(flows, dtype=np.float32)


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(x, kernel, mode="same")


def estimate_cycle_lengths(
    time_array: np.ndarray,
    flow_array: np.ndarray,
    threshold_lpm: float = 2.0,
    smooth_width: int = 21,
    min_phase_sec: float = 0.6,
    min_cycle_sec: float = 2.0,
    max_cycle_sec: float = 6.0,
) -> dict[str, object]:
    smoothed = moving_average(flow_array, smooth_width)
    state = np.zeros_like(smoothed, dtype=np.int8)
    state[smoothed >= threshold_lpm] = 1
    state[smoothed <= -threshold_lpm] = -1

    transitions: list[tuple[float, int]] = []
    current_state = state[0]
    current_start = float(time_array[0])
    for idx in range(1, len(state)):
        if state[idx] == current_state:
            continue
        segment_end = float(time_array[idx])
        duration = segment_end - current_start
        if current_state != 0 and duration >= min_phase_sec:
            transitions.append((current_start, int(current_state)))
        current_state = int(state[idx])
        current_start = float(time_array[idx])
    last_duration = float(time_array[-1]) - current_start
    if current_state != 0 and last_duration >= min_phase_sec:
        transitions.append((current_start, int(current_state)))

    cycle_lengths = []
    insp_lengths = []
    exp_lengths = []
    for idx in range(len(transitions) - 2):
        start_t, start_state = transitions[idx]
        mid_t, mid_state = transitions[idx + 1]
        end_t, end_state = transitions[idx + 2]
        if start_state == 1 and mid_state == -1 and end_state == 1:
            insp = mid_t - start_t
            exp = end_t - mid_t
            cycle = end_t - start_t
            if min_cycle_sec <= cycle <= max_cycle_sec:
                cycle_lengths.append(cycle)
                insp_lengths.append(insp)
                exp_lengths.append(exp)

    return {
        "num_cycles": len(cycle_lengths),
        "cycle_lengths_sec": cycle_lengths,
        "insp_lengths_sec": insp_lengths,
        "exp_lengths_sec": exp_lengths,
    }


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate breathing cycle statistics from flow signals.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="summary-MMmodel/breath_cycle_analysis.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    session_results = []
    all_cycles: list[float] = []
    all_insp: list[float] = []
    all_exp: list[float] = []

    for session_dir in sorted(data_root.glob("MMdata_*")):
        daq_path = session_dir / "daq.csv"
        if not daq_path.exists():
            continue
        time_array, flow_array = load_flow(daq_path)
        stats = estimate_cycle_lengths(time_array, flow_array)
        cycle_lengths = stats["cycle_lengths_sec"]
        insp_lengths = stats["insp_lengths_sec"]
        exp_lengths = stats["exp_lengths_sec"]
        session_results.append(
            {
                "session_id": session_dir.name,
                "num_cycles": stats["num_cycles"],
                "cycle_summary": summarize(cycle_lengths),
                "insp_summary": summarize(insp_lengths),
                "exp_summary": summarize(exp_lengths),
            }
        )
        all_cycles.extend(cycle_lengths)
        all_insp.extend(insp_lengths)
        all_exp.extend(exp_lengths)

    output = {
        "num_sessions": len(session_results),
        "global_cycle_summary": summarize(all_cycles),
        "global_insp_summary": summarize(all_insp),
        "global_exp_summary": summarize(all_exp),
        "session_results": session_results,
        "recommended_window_sec_candidates": {
            "cycle_median": median(all_cycles) if all_cycles else None,
            "cycle_p10_p90": [float(np.percentile(all_cycles, 10)), float(np.percentile(all_cycles, 90))]
            if all_cycles
            else None,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(output["global_cycle_summary"], ensure_ascii=False))
    print(json.dumps(output["global_insp_summary"], ensure_ascii=False))
    print(json.dumps(output["global_exp_summary"], ensure_ascii=False))
    print(output_path)


if __name__ == "__main__":
    main()
