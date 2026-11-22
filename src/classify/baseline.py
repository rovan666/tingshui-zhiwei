# -*- coding: utf-8 -*-
"""同数据 log-Mel 逻辑回归基线，用于判断 TinySwin 是否确有必要。"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.logmel import LogMelExtractor


class LogMelLogisticBaseline:
    """八个时间块保留调频/脉冲包络，避免小样本直接展开数万个像素。"""
    def __init__(self, extractor: LogMelExtractor):
        self.extractor = extractor
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=500, solver="lbfgs")

    def _features(self, wave):
        feature = self.extractor.transform(wave)[0]
        blocks = [block.mean(axis=1) for block in np.array_split(feature, 8, axis=1)]
        return np.concatenate(blocks + [feature.std(axis=1)])

    def fit(self, waves, labels):
        features = np.stack([self._features(w) for w in waves])
        self.model.fit(self.scaler.fit_transform(features), labels)
        return self

    def predict_proba(self, wave):
        features = self.scaler.transform(self._features(wave)[None])
        raw = self.model.predict_proba(features)[0]
        ordered = np.zeros(3)
        for index, label in enumerate(self.model.classes_):
            ordered["ABC".index(str(label))] = raw[index]
        return ordered
