# -*- coding: utf-8 -*-
"""多通道 WAV、事件标签和波形哈希的持久化契约。"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import wave

import numpy as np


@dataclass
class EventLabel:
    source_id: str
    class_id: str
    start_sample: int
    end_sample: int
    x_m: float
    y_m: float
    theta_deg: float
    snr_db: float
    seed: int
    mode: str
    config_id: str

    def __post_init__(self):
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("事件区间必须为右开正区间")
        if self.class_id not in {"A", "B", "C", "unknown"}:
            raise ValueError("未知类别标签")


@dataclass
class Session:
    session_id: str
    fs: int
    audio: np.ndarray
    events: list = field(default_factory=list)
    mode: str = "simulation"
    config_id: str = "water_ula16_v1"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.audio = np.asarray(self.audio, dtype=np.float32)
        if self.audio.ndim != 2 or self.audio.shape[0] != 16:
            raise ValueError("会话波形必须为 [16,N]")
        for event in self.events:
            if event.end_sample > self.audio.shape[1]:
                raise ValueError("事件标签超出波形长度")


def save_session(session: Session, directory) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    wav_path = directory / f"{session.session_id}.wav"
    json_path = directory / f"{session.session_id}.json"
    scale = max(1.0, float(np.max(np.abs(session.audio))))
    pcm = np.round(np.clip(session.audio / scale, -1, 1) * 32767).astype("<i2")
    interleaved = pcm.T.reshape(-1).tobytes()
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(16)
        wf.setsampwidth(2)
        wf.setframerate(session.fs)
        wf.writeframes(interleaved)
    metadata = dict(session.metadata)
    metadata["pcm_scale"] = scale
    payload = {
        "session_id": session.session_id, "fs": session.fs,
        "wav": wav_path.name, "mode": session.mode, "config_id": session.config_id,
        "events": [asdict(e) for e in session.events], "metadata": metadata,
        "waveform_sha256": hashlib.sha256(interleaved).hexdigest(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return wav_path, json_path


def load_session(path) -> Session:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    wav_path = path.parent / payload["wav"]
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getnchannels() != 16 or wf.getsampwidth() != 2:
            raise ValueError("会话 WAV 必须为 16 通道 PCM16")
        fs = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if hashlib.sha256(raw).hexdigest() != payload["waveform_sha256"]:
        raise ValueError("WAV 波形哈希不匹配")
    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, 16).T
    audio = pcm.astype(np.float32) / 32767.0 * float(payload["metadata"]["pcm_scale"])
    if fs != payload["fs"]:
        raise ValueError("WAV 与标注采样率不一致")
    return Session(
        session_id=payload["session_id"], fs=fs, audio=audio,
        events=[EventLabel(**e) for e in payload["events"]], mode=payload["mode"],
        config_id=payload["config_id"], metadata=payload["metadata"],
    )
