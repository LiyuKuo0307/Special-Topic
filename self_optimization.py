# -*- coding: utf-8 -*-
"""
自我診斷／自我評估／自我優化三層機制（對應第七節）。
以「執行紀錄」為核心：每次任務記錄採用策略與結果，據此校準信心、優化策略、產出自我優化報告。
執行紀錄用 JSON 檔落地（預設 data/agent_runs.json），可換成 SQLite。
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from .llm_client import loads_llm_json

DEFAULT_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "agent_runs.json")


# ---------- (一) 自我診斷（第七節一） ----------
def diagnose_crawler(rows: list) -> dict:
    """爬蟲失敗或回空 → 判斷錯誤類型並建議動作（備援選擇器／重試／切換來源）。"""
    errs = [r for r in rows if isinstance(r, dict) and r.get("_crawler_error")]
    if errs:
        return {"ok": False, "issue": errs[0]["_crawler_error"],
                "action": "切換備援資料來源或延遲重試", "auto_resolved": False}
    if not rows:
        return {"ok": False, "issue": "回傳資料為空", "action": "擴展關鍵字或改用備援來源",
                "auto_resolved": False}
    return {"ok": True}


def diagnose_llm_json(raw: str) -> dict:
    """LLM 回傳格式錯誤／JSON 解析失敗 → 建議附加格式修正提示或降級輸出。"""
    try:
        loads_llm_json(raw)
        return {"ok": True}
    except Exception:
        return {"ok": False, "issue": "LLM 輸出非合法 JSON",
                "action": "附加格式修正提示重送；連續失敗則降級為簡單輸出", "auto_resolved": True}


def diagnose_evidence_conflict(fact_checks: list) -> dict:
    """多來源證據彼此矛盾 → 觸發第二輪交叉比對；無法收斂則標證據不足。"""
    labels = {f.get("credibility_label") for f in fact_checks if f}
    if len(labels) > 1:
        return {"ok": False, "issue": f"多來源判定不一致：{labels}",
                "action": "交叉比對；仍分歧則標註證據不足", "auto_resolved": True}
    return {"ok": True}


# ---------- (二) 自我評估（第七節二） ----------
def consistency_check(results: list) -> dict:
    """一致性檢查：多次重跑結果是否收斂（以可信度標籤是否一致衡量）。"""
    labels = [r.get("fact_check", {}).get("credibility_label") for r in results if r.get("fact_check")]
    if not labels:
        return {"consistent": True, "note": "無可比對結果"}
    most = max(set(labels), key=labels.count)
    agreement = labels.count(most) / len(labels)
    return {"consistent": agreement >= 0.6, "agreement": round(agreement, 2),
            "dominant_label": most,
            "note": "結果分歧過大，建議判低信心並重跑" if agreement < 0.6 else "結果收斂"}


def evidence_sufficiency_check(fact_check: dict) -> dict:
    """證據充分性檢查：高信心判定是否有實際佐證支撐。"""
    from .schema import CONFIDENT_LABELS
    if not fact_check:
        return {"ok": True}
    high_conf = fact_check.get("confidence", 0) >= 0.6 and fact_check.get("credibility_label") in CONFIDENT_LABELS
    if high_conf and not fact_check.get("evidence_sufficient"):
        return {"ok": False, "issue": "高信心判定但證據不足", "action": "調降信心或標待複核"}
    return {"ok": True}


# ---------- (三) 自我優化（第七節三） ----------
class RunLog:
    """執行紀錄存取。記錄策略與成敗，供策略成效追蹤與信心校準。"""

    def __init__(self, path: str | None = None):
        self.path = path or DEFAULT_LOG
        try:
            with open(self.path, encoding="utf-8") as f:
                self.runs = json.load(f)
        except Exception:
            self.runs = []

    def record(self, strategy: str, success: bool, task_type: str = "fact_check",
               confidence: float = 0.0, correct: bool | None = None):
        self.runs.append({"strategy": strategy, "success": success, "task_type": task_type,
                          "confidence": confidence, "correct": correct})

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.runs, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def strategy_success_rates(self) -> dict:
        """策略成效追蹤：各策略成功率，供提高高成效策略權重。"""
        agg = defaultdict(lambda: [0, 0])
        for r in self.runs:
            agg[r["strategy"]][1] += 1
            if r["success"]:
                agg[r["strategy"]][0] += 1
        return {s: round(ok / n, 2) for s, (ok, n) in agg.items() if n}

    def best_strategy(self) -> str | None:
        rates = self.strategy_success_rates()
        return max(rates, key=rates.get) if rates else None

    def confidence_calibration(self, task_type: str = "fact_check") -> dict:
        """信心校準：比對自評信心與歷史正確率，長期不符則建議調降信心權重。"""
        rel = [r for r in self.runs if r["task_type"] == task_type and r.get("correct") is not None]
        if not rel:
            return {"calibrated": True, "note": "尚無標註正確性的歷史資料"}
        avg_conf = sum(r["confidence"] for r in rel) / len(rel)
        acc = sum(1 for r in rel if r["correct"]) / len(rel)
        gap = round(avg_conf - acc, 2)
        return {"calibrated": abs(gap) <= 0.15, "avg_confidence": round(avg_conf, 2),
                "accuracy": round(acc, 2), "gap": gap,
                "suggest": "調降該類任務信心權重" if gap > 0.15 else "信心與準確率大致相符"}


def periodic_report(run_log: RunLog) -> dict:
    """定期彙總報告（第七節三）：摘要策略成效與信心校準，供人力必要時審視。"""
    return {
        "total_runs": len(run_log.runs),
        "strategy_success_rates": run_log.strategy_success_rates(),
        "best_strategy": run_log.best_strategy(),
        "confidence_calibration": run_log.confidence_calibration(),
    }
