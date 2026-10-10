# -*- coding: utf-8 -*-
"""通用小工具：目录、JSON 落盘、日志。"""

import json
from pathlib import Path


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return path


def load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
