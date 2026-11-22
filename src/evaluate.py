# -*- coding: utf-8 -*-
"""事件级匹配与仿真/实录分离的验收指标。"""
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score


def match_events(predictions, labels):
    predictions = sorted(predictions, key=lambda p: p["time_s"])
    unmatched = set(range(len(labels)))
    matched, false = [], []
    for pred in predictions:
        candidates = [i for i in unmatched
                      if labels[i]["start_s"] <= pred["time_s"] <= labels[i]["end_s"] + 0.5]
        if not candidates:
            false.append(pred)
            continue
        label_index = min(candidates, key=lambda i: labels[i]["start_s"])
        unmatched.remove(label_index)
        matched.append((pred, labels[label_index]))
    missed = [label for i, label in enumerate(labels) if i in unmatched]
    return matched, missed, false


def _metrics(records):
    all_matches, all_missed, all_false = [], [], []
    background_minutes = 0.0
    background_false = 0
    for record in records:
        matched, missed, false = match_events(record["predictions"], record["labels"])
        all_matches.extend(matched)
        all_missed.extend(missed)
        all_false.extend(false)
        background_minutes += float(record.get("background_minutes", 0.0))
        background_false += int(record.get("background_false_alarms", 0))
    truth_count = len(all_matches) + len(all_missed)
    valid = [(p, l) for p, l in all_matches if p.get("angle_deg") is not None]
    errors = np.array([abs(float(p["angle_deg"]) - float(l["theta_deg"])) for p, l in valid])
    known_truth, known_pred = [], []
    unknown_accepted = unknown_total = 0
    delays = []
    for pred, label in all_matches:
        delays.append(float(pred["time_s"] - label["start_s"]))
        if label["class_id"] in {"A", "B", "C"}:
            known_truth.append(label["class_id"])
            known_pred.append(pred.get("label", "unknown"))
        elif label["class_id"] == "unknown":
            unknown_total += 1
            unknown_accepted += int(pred.get("label") in {"A", "B", "C"})
    matched_count = len(all_matches)
    return {
        "event_recall": len(all_matches) / max(truth_count, 1),
        "missed_events": len(all_missed),
        "extra_false_alarms": len(all_false),
        "background_false_alarms_per_minute": background_false / max(background_minutes, 1e-12),
        "background_minutes": background_minutes,
        "angle_mae_deg": float(np.mean(errors)) if len(errors) else None,
        "angle_p90_deg": float(np.percentile(errors, 90)) if len(errors) else None,
        "angle_coverage": len(valid) / max(matched_count, 1),
        "bearing_rejection_rate": (matched_count - len(valid)) / max(matched_count, 1),
        "bearing_rejections": matched_count - len(valid),
        "macro_f1": float(f1_score(known_truth, known_pred, labels=["A", "B", "C"],
                                   average="macro", zero_division=0)) if known_truth else None,
        "unknown_false_acceptance": unknown_accepted / max(unknown_total, 1) if unknown_total else None,
        "mean_trigger_delay_s": float(np.mean(delays)) if delays else None,
    }


def evaluate_sessions(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["mode"]].append(record)
    return {"by_mode": {mode: _metrics(items) for mode, items in grouped.items()}}
