# -*- coding: utf-8 -*-
"""輕量除錯日誌：把查核過程的原始 LLM 回應與錯誤寫到 logs/run.log（JSON Lines）。
用途：真實模式出現怪結果時，直接看這個檔就知道 LLM 到底回了什麼、哪裡失敗——不用截圖。
best-effort：寫檔失敗絕不影響主流程。用環境變數 MISINFO_DEBUG=0 可關閉。"""
from __future__ import annotations
import json
import os
import datetime

_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
_LOG_PATH = os.path.join(_LOG_DIR, "run.log")


def enabled() -> bool:
    return os.environ.get("MISINFO_DEBUG", "1") != "0"


def log_event(event: str, **data):
    if not enabled():
        return
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "event": event}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 2000:
                v = v[:2000] + "…(截斷)"
            rec[k] = v
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 日誌絕不能影響主流程
