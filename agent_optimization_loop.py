# -*- coding: utf-8 -*-
"""
AI Agent 自我優化迴圈（對應企劃書第九節 D + 第五節「Agent 自我評估層」+ 第十四節完成定義）：
  自我診斷（資料品質）→ 報告產出 → LLM-as-Judge 自我評分 → 未達標則帶回饋重試/精修。

兩層重試要分清楚，不要混在一起改：
- 本檔：內容『品質』層——報告寫得夠不夠具體、可信度判斷有沒有依據。
- claude_client.py：API『連線』層——逾時、限流、金鑰錯誤等網路層問題。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .data_pipeline import clean_comments
from .llm_client import LLMClient, MARK_JUDGE, MARK_REPORT, _validate_judge_json
from .schema import EVAL_KEYS


# ---------- 1. 自我診斷：資料品質（對應 6.2 有效留言比例判定） ----------

@dataclass
class DataQualityResult:
    passed: bool
    total_rows: int
    valid_rows: int
    valid_ratio: float
    issues: list


def check_data_quality(rows: list, cleaned_rows: list,
                       min_valid_ratio: float = 0.5, min_rows: int = 8) -> DataQualityResult:
    """
    分開檢查兩件事，不能混看同一個數字：
    - valid_ratio：原始資料中非空白（不是垃圾）的比例——抓「原始資料太髒」。
    - usable：清洗＋去重後還剩幾筆不重複留言——抓「重複洗版、實際可分析內容太少」。
    """
    issues = []
    total = len(rows)
    valid = len([r for r in rows if str(r.get("content") or "").strip()])   # 容錯：非字串（第十七節）
    ratio = (valid / total) if total else 0.0
    usable = len(cleaned_rows)

    if usable < min_rows:
        issues.append(f"清洗去重後可用留言僅 {usable} 筆（建議至少 {min_rows} 筆不重複留言）")
    if ratio < min_valid_ratio:
        issues.append(f"原始資料中有效（非空白）比例過低（{ratio:.0%}，門檻 {min_valid_ratio:.0%}）")

    return DataQualityResult(
        passed=len(issues) == 0,
        total_rows=total,
        valid_rows=usable,
        valid_ratio=round(ratio, 3),
        issues=issues,
    )


# ---------- 2. 自我評估（LLM-as-Judge，對應 6.3） ----------

def build_judge_prompt(report: str) -> str:
    return (
        f"{MARK_JUDGE}\n"
        "請你擔任評分者，依下列四個指標為這份假訊息查核報告打 1-5 分，並只以 JSON 回覆：\n"
        f"報告內容：{report}\n"
        "指標：specificity（具體性）、readability（可讀性）、relevance（與待查核主張的對應性）、"
        "risk_reasoning（可信度判斷是否有證據依據）。\n"
        "另附 feedback 欄位，用一句話說明最該改進之處。"
    )


def self_evaluate(llm: LLMClient, report: str, threshold: int = 3) -> dict:
    raw = llm.generate(build_judge_prompt(report))
    scores = _validate_judge_json(raw)
    scores["passed"] = all(int(scores.get(k, 0) or 0) >= threshold for k in EVAL_KEYS)
    return scores


# ---------- 3. 自我優化迴圈（對應 6.4） ----------

def build_report_prompt(topic: str, analysis: dict, fact_check: dict,
                        feedback: Optional[str] = None) -> str:
    import json as _json
    base = (
        f"{MARK_REPORT}\n"
        "請根據以下待查核主題、統計數據與查核結果，撰寫一份具體、可讀、有證據依據的"
        "假訊息查核報告。需點出可信度標籤與判斷依據，並引用至少一則代表留言。\n"
        f"主題：{topic}\n"
        f"統計數據：{_json.dumps(analysis, ensure_ascii=False)}\n"
        f"查核結果：{_json.dumps(fact_check, ensure_ascii=False)}"
    )
    if feedback:
        base += f"\n\n上一版報告的問題：{feedback}\n請務必針對這個問題修正。"
    return base


def generate_report_with_optimization(
    llm: LLMClient,
    topic: str,
    analysis: dict,
    fact_check: dict,
    max_attempts: int = 3,
    threshold: int = 3,
) -> dict:
    """未達標時把評分回饋帶進下一輪重新生成；達重試上限仍未過，標記人工複核。"""
    attempts_log = []
    feedback = None
    final_report = None

    for attempt in range(1, max_attempts + 1):
        prompt = build_report_prompt(topic, analysis, fact_check, feedback)
        report = llm.generate(prompt)
        eval_result = self_evaluate(llm, report, threshold)

        attempts_log.append({
            "attempt": attempt,
            "report": report,
            **{k: eval_result.get(k) for k in EVAL_KEYS},
            "passed": eval_result["passed"],
            "feedback": eval_result.get("feedback", ""),
        })

        final_report = report
        if eval_result["passed"]:
            break
        feedback = eval_result.get("feedback", "")

    last = attempts_log[-1]
    # 綜合四項給一個 0-10 的 judge_score，方便 Dashboard 一眼看整體品質（對應企劃書 agent_self_evaluation.judge_score）
    judge_avg = sum(int(last.get(k, 0) or 0) for k in EVAL_KEYS) / len(EVAL_KEYS)
    judge_score = round(judge_avg / 5 * 10, 1)

    return {
        "ai_report": final_report,
        "agent_self_evaluation": {
            "judge_score": judge_score,
            "passed_threshold": last["passed"],
            "retry_count": len(attempts_log) - 1,
            "notes": last.get("feedback", ""),
        },
        "agent_log": {
            "attempts": len(attempts_log),
            "self_evaluation": attempts_log,
            "human_review_required": not last["passed"],
        },
    }
