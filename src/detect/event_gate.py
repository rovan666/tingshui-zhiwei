# -*- coding: utf-8 -*-
"""OS-SCL 在线双阈值状态机、2 秒同步回溯与验证集标定。"""
from collections import deque
import numpy as np


class SyncRingBuffer:
    def __init__(self, fs: int = 100_000, seconds: float = 2.0):
        self.capacity = int(round(fs * seconds))
        self._chunks = []
        self.size = 0

    def append(self, chunk: np.ndarray):
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] != 16:
            raise ValueError("回溯输入必须为 [16,N]")
        self._chunks.append(chunk)
        self.size += chunk.shape[1]
        if self.size > self.capacity:
            joined = np.concatenate(self._chunks, axis=1)[:, -self.capacity:]
            self._chunks = [joined]
            self.size = joined.shape[1]

    def get(self) -> np.ndarray:
        if not self._chunks:
            return np.empty((16, 0), dtype=np.float32)
        return np.concatenate(self._chunks, axis=1)[:, -self.capacity:].copy()

    def clear(self):
        self._chunks.clear()
        self.size = 0


class EventGate:
    def __init__(self, tau_on: float, tau_off: float, fs: int = 100_000,
                 prebuffer_s: float = 2.0, update_s: float = 0.25):
        if not tau_off < tau_on:
            raise ValueError("关阈值必须低于开阈值")
        self.tau_on = float(tau_on)
        self.tau_off = float(tau_off)
        self.update_s = update_s
        self.ring = SyncRingBuffer(fs, prebuffer_s)
        self.recent = deque(maxlen=3)
        self.active = False
        self.low_count = 0
        self.seq = 0
        self.next_event_id = 1
        self.current_event_id = None

    def reset(self):
        self.recent.clear()
        self.ring.clear()
        self.active = False
        self.low_count = 0
        self.current_event_id = None

    def update(self, score: float, chunk: np.ndarray = None,
               acquisition_error: bool = False) -> dict:
        self.seq += 1
        result = {
            "seq": self.seq, "score": float(score), "trigger": False, "close": False,
            "active": False, "event_id": None, "status": "listening",
            "replay_audio": None,
        }
        if acquisition_error:
            self.reset()
            result["status"] = "acquisition_error"
            return result
        if chunk is not None:
            self.ring.append(chunk)
        score = float(score)
        self.recent.append(score)
        if not self.active:
            if len(self.recent) == 3 and sum(s > self.tau_on for s in self.recent) >= 2:
                self.active = True
                self.low_count = 0
                self.current_event_id = self.next_event_id
                self.next_event_id += 1
                result.update(trigger=True, replay_audio=self.ring.get())
        else:
            self.low_count = self.low_count + 1 if score < self.tau_off else 0
            if self.low_count >= 3:
                result.update(close=True, event_id=self.current_event_id)
                self.active = False
                self.low_count = 0
                self.current_event_id = None
                return result
        if self.active:
            result.update(active=True, event_id=self.current_event_id, status="event_active")
        return result


def _replay_scores(gate: EventGate, scores):
    triggers, first = 0, None
    for index, score in enumerate(scores):
        result = gate.update(float(score))
        if result["trigger"]:
            triggers += 1
            if first is None:
                first = index
    return triggers, first


def calibrate_gate(background_sessions, event_sessions) -> dict:
    backgrounds = [np.asarray(s, dtype=float) for s in background_sessions]
    events = [np.asarray(s, dtype=float) for s in event_sessions]
    if not backgrounds or not events:
        raise ValueError("阈值标定需要纯背景和含事件验证会话")
    background_scores = np.concatenate(backgrounds)
    total_minutes = sum(len(s) for s in backgrounds) * 0.25 / 60
    candidates = []
    for on_q in (90, 95, 97.5, 99):
        tau_on = float(np.percentile(background_scores, on_q))
        for off_q in (80, 90):
            tau_off = float(np.percentile(background_scores, off_q))
            if not tau_off < tau_on:
                continue
            false_triggers = 0
            for session in backgrounds:
                false_triggers += _replay_scores(EventGate(tau_on, tau_off), session)[0]
            recalls, delays = [], []
            for session in events:
                count, first = _replay_scores(EventGate(tau_on, tau_off), session)
                recalls.append(float(count > 0))
                if first is not None:
                    delays.append(first * 0.25)
            candidates.append({
                "tau_on": tau_on, "tau_off": tau_off,
                "false_alarms_per_minute": false_triggers / max(total_minutes, 1e-12),
                "event_recall": float(np.mean(recalls)),
                "mean_delay_s": float(np.mean(delays)) if delays else float("inf"),
            })
    if not candidates:
        raise ValueError("背景分位数无法产生 tau_off < tau_on 的候选")
    valid = [c for c in candidates if c["false_alarms_per_minute"] <= 1.0]
    pool = valid if valid else candidates
    if valid:
        best = min(pool, key=lambda c: (-c["event_recall"], c["mean_delay_s"],
                                        c["false_alarms_per_minute"]))
    else:
        best = min(pool, key=lambda c: (c["false_alarms_per_minute"],
                                        -c["event_recall"], c["mean_delay_s"]))
    best["target_met"] = bool(valid)
    best["state_machine_version"] = "2of3_on_3low_off_v1"
    best["candidate_count"] = len(candidates)
    return best
