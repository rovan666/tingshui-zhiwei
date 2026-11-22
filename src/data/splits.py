# -*- coding: utf-8 -*-
"""先按母素材/会话分组，再划分训练、验证和测试。"""
from collections import defaultdict

import numpy as np


def split_master_ids(ids, seed: int = 0) -> dict[str, list[str]]:
    if isinstance(ids, dict):
        groups = {k: list(v) for k, v in ids.items()}
    else:
        groups = defaultdict(list)
        for source_id in ids:
            groups[str(source_id).split("_")[0]].append(str(source_id))
    rng = np.random.default_rng(seed)
    result = {"train": [], "val": [], "test": []}
    for _, values in sorted(groups.items()):
        shuffled = list(rng.permutation(values))
        n_train = int(round(0.6 * len(shuffled)))
        n_val = int(round(0.2 * len(shuffled)))
        result["train"].extend(shuffled[:n_train])
        result["val"].extend(shuffled[n_train:n_train + n_val])
        result["test"].extend(shuffled[n_train + n_val:])
    for values in result.values():
        values.sort()
    flat = [x for values in result.values() for x in values]
    if len(flat) != len(set(flat)):
        raise RuntimeError("数据划分出现母素材泄漏")
    return result


def assert_session_partition(session_ids, source_to_split: dict[str, str]) -> None:
    """同一会话及其全部事件只能归属一个集合。"""
    seen = {}
    for session_id, source_id in session_ids:
        split = source_to_split[source_id]
        if session_id in seen and seen[session_id] != split:
            raise ValueError("同一录制会话跨集合")
        seen[session_id] = split
