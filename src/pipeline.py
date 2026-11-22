# -*- coding: utf-8 -*-
"""以 session_id/seq/event_id 串联检测、测向、分类和事件日志。"""
import time
import numpy as np


class InferenceEngine:
    def __init__(self, cfg, detector, gate, doa, classifier,
                 config_hash: str = "", model_versions=None):
        self.cfg = cfg
        self.detector = detector
        self.gate = gate
        self.doa = doa
        self.classifier = classifier
        self.config_hash = config_hash
        self.model_versions = model_versions or {}
        self.event_log = []
        self.last_replay_audio = None
        self._stream_session = None
        self._stream_next_seq = None

    def process_window(self, window: np.ndarray, session_id: str, seq: int,
                       acquisition_error: bool = False) -> dict:
        started = time.perf_counter()
        sequential = (self._stream_session == session_id and self._stream_next_seq == seq)
        if not sequential:
            self.gate.reset()
            for component in (self.doa, self.classifier):
                if hasattr(component, "reset"):
                    component.reset()
        score = 0.0 if acquisition_error else float(self.detector(window))
        update_samples = self.cfg.features.update_samples
        chunk = window if not sequential else window[:, -min(update_samples, window.shape[1]):]
        gated = self.gate.update(score, None if acquisition_error else chunk,
                                 acquisition_error=acquisition_error)
        self._stream_session = session_id
        self._stream_next_seq = None if acquisition_error else seq + 1
        if gated["trigger"]:
            self.last_replay_audio = gated["replay_audio"]
        result = {
            "session_id": session_id, "seq": seq, "event_id": gated["event_id"],
            "time_s": seq * self.cfg.features.update_samples / self.cfg.array.fs,
            "status": gated["status"], "trigger": gated["trigger"], "close": gated["close"],
            "detection_score": score, "angle_deg": None, "point_index": None,
            "label": "unknown", "probability": 0.0, "reason": gated["status"],
            "config_hash": self.config_hash, "model_versions": dict(self.model_versions),
        }
        if acquisition_error:
            for component in (self.doa, self.classifier):
                if hasattr(component, "reset"):
                    component.reset()
            result["elapsed_ms"] = (time.perf_counter() - started) * 1000
            self.event_log.append(result.copy())
            return result
        if gated["trigger"] and hasattr(self.classifier, "reset"):
            self.classifier.reset()
        if gated["trigger"] or gated["active"]:
            bearing = self.doa(window)
            result.update(
                angle_deg=bearing.get("angle_deg") if bearing.get("valid") else None,
                point_index=bearing.get("point_index"), reason=bearing.get("reason", "ok"),
                peak_ratio=bearing.get("peak_ratio"),
            )
            classified = self.classifier(window, bearing, score)
            if classified:
                result.update(label=classified.get("label", "unknown"),
                              probability=float(classified.get("probability", 0.0)),
                              classification_pending=bool(classified.get("classification_pending", False)))
        elif gated["close"] and hasattr(self.classifier, "close"):
            classified = self.classifier.close()
            if classified:
                result.update(label=classified.get("label", "unknown"),
                              probability=float(classified.get("probability", 0.0)),
                              classification_pending=False)
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        self.event_log.append(result.copy())
        return result


def replay_session(session, engine: InferenceEngine):
    # 会话间清空门控、回溯与事件分类状态，但保留 event_id 的全局递增编号。
    engine.gate.reset()
    for component in (engine.doa, engine.classifier):
        if hasattr(component, "reset"):
            component.reset()
    n = session.audio.shape[1]
    window_samples = engine.cfg.features.window_samples
    update_samples = engine.cfg.features.update_samples
    if n < window_samples:
        audio = np.pad(session.audio, ((0, 0), (0, window_samples - n)))
    else:
        audio = session.audio
    results = []
    for seq, start in enumerate(range(0, audio.shape[1] - window_samples + 1, update_samples)):
        window = audio[:, start:start + window_samples]
        results.append(engine.process_window(window, session.session_id, seq))
    return results
