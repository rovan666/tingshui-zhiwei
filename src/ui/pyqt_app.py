# -*- coding: utf-8 -*-
"""PyQt5 单屏事件界面：波形、事件状态、方位射线和 A/B/C/未知。"""
import math
import os
from pathlib import Path
import numpy as np

# Qt5 在 Windows 中文绝对插件路径下可能阻塞；项目根目录运行时使用 ASCII 相对路径。
_plugin_dir = Path(__file__).resolve().parents[2] / ".venv/Lib/site-packages/PyQt5/Qt5/plugins/platforms"
if _plugin_dir.is_dir():
    _relative_plugins = os.path.relpath(_plugin_dir, Path.cwd())
    if _relative_plugins.isascii():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _relative_plugins)

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget


class WaveformWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.waveform = np.zeros(0)
        self.setMinimumHeight(180)

    def set_waveform(self, waveform):
        self.waveform = np.asarray(waveform)[7]
        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if not len(self.waveform):
            return
        painter.setPen(QPen(Qt.cyan, 1))
        indices = np.linspace(0, len(self.waveform) - 1, min(self.width(), len(self.waveform))).astype(int)
        values = self.waveform[indices]
        scale = max(float(np.max(np.abs(values))), 1e-9)
        points = [QPointF(i * self.width() / max(len(values) - 1, 1),
                          self.height() / 2 - values[i] / scale * self.height() * 0.45)
                  for i in range(len(values))]
        for a, b in zip(points[:-1], points[1:]):
            painter.drawLine(a, b)


class BearingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.angle = None
        self.setMinimumHeight(180)

    def set_angle(self, angle):
        self.angle = angle
        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.white)
        center = QPointF(self.width() / 2, self.height() * 0.9)
        painter.setPen(QPen(Qt.gray, 1))
        painter.drawLine(0, int(center.y()), self.width(), int(center.y()))
        if self.angle is None:
            return
        radians = math.radians(self.angle)
        end = QPointF(center.x() + math.sin(radians) * self.height() * 0.75,
                      center.y() - math.cos(radians) * self.height() * 0.75)
        painter.setPen(QPen(Qt.red, 3))
        painter.drawLine(center, end)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("听水·知微｜16 通道线阵水声监测")
        self.session_label = QLabel("—")
        self.status_label = QLabel("监听中")
        self.bearing_label = QLabel("方位不确定")
        self.class_label = QLabel("未知")
        self.reason_label = QLabel("原因：—")
        self.waveform = WaveformWidget()
        self.bearing = BearingWidget()
        layout = QVBoxLayout()
        for widget in (self.session_label, self.status_label, self.bearing_label,
                       self.class_label, self.reason_label, self.waveform, self.bearing):
            layout.addWidget(widget)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.resize(900, 700)

    def set_event(self, event):
        self.session_label.setText(f"{event['session_id']} / {event['seq']} / {event['event_id']}")
        status = {"event_active": "事件进行中", "acquisition_error": "采集异常",
                  "listening": "监听中"}.get(event["status"], event["status"])
        self.status_label.setText(status)
        angle = event.get("angle_deg")
        self.bearing_label.setText(f"{angle:.1f}°" if angle is not None else "方位不确定")
        self.class_label.setText("未知" if event.get("label") == "unknown" else event.get("label", "未知"))
        self.reason_label.setText(f"原因：{event.get('reason', '—')}")
        self.waveform.set_waveform(event.get("waveform", np.zeros((16, 0))))
        self.bearing.set_angle(angle)


def launch(payload=None):
    """启动事件窗口；只显示传入的真实回放结果，不生成假告警。"""
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    payload = payload or {}
    events = payload.get("results", [])
    if not events and payload.get("sessions"):
        events = []
        for session in payload["sessions"]:
            for alarm in session.get("alarms", []):
                events.append({"session_id": session["session_id"], "seq": alarm["alarm_index"],
                               "event_id": alarm["alarm_index"] + 1, "status": "event_active",
                               "angle_deg": alarm["bearing"]["angle_deg"],
                               "label": alarm["predicted_class"],
                               "probability": max(alarm["class_probabilities"]),
                               "reason": alarm["bearing"]["reason"]})
    if events:
        window.set_event(events[-1])
    window.show()
    return app.exec_()


if __name__ == "__main__":
    launch()
