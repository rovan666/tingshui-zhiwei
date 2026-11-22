# -*- coding: utf-8 -*-
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH",
                      ".venv/Lib/site-packages/PyQt5/Qt5/plugins/platforms")
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    QApplication = None


@unittest.skipIf(QApplication is None, "项目虚拟环境提供 PyQt5 后运行")
class ULA16UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()

    def test_main_window_displays_event_status_bearing_and_unknown(self):
        import numpy as np
        from src.ui.pyqt_app import MainWindow
        window = MainWindow()
        try:
            window.set_event({
                "session_id": "s01", "seq": 3, "event_id": 2, "status": "event_active",
                "angle_deg": 30.0, "label": "unknown", "probability": 0.42,
                "waveform": np.zeros((16, 1000)), "reason": "low_confidence",
            })
            self.assertEqual(window.session_label.text(), "s01 / 3 / 2")
            self.assertEqual(window.status_label.text(), "事件进行中")
            self.assertEqual(window.bearing_label.text(), "30.0°")
            self.assertEqual(window.class_label.text(), "未知")
            self.assertIn("low_confidence", window.reason_label.text())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
