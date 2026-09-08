# -*- coding: utf-8 -*-
"""
LLM 查核與可信度評估（對應企劃書第九節 D + 第八節 fact_check 欄位）。

流程：把「待查核主張 + 可疑留言 + 可比對證據線索」組成 prompt 丟給 LLM，
要求它回傳結構化 JSON（claim / credibility_label / confidence / evidence_summary /
evidence_sufficient），再由本模組做「護欄」後處理。

護欄（對應企劃書第十五節「查核報告太空泛或武斷」的解法）：
- 標籤必須是 schema.CREDIBILITY_LABELS 之一，否則降級為「待人工複核」。
- 證據不足時，不得輸出高信心的真/偽判定：強制把標籤改為「證據不足」並壓低 confidence。
  這正是企劃書反覆強調的「保留證據不足選項、避免武斷判定」。
"""

from __future__ import annotations

import json

from llm_client import LLMClient, MARK_FACT_CHECK
from schema import (
    CREDIBILITY_LABELS,
    CREDIBILITY_INSUFFICIENT,
    CREDIBILITY_NEEDS_REVIEW,
    CONFIDENT_LABELS,
)

# 證據不足時，confidence 不得超過這個上限（避免「沒證據卻很有信心」的矛盾輸出）。
INSUFFICIENT_CONFIDENCE_CAP = 0.4


def build_fact_check_prompt(topic: str, suspicious_comments: list, evidence_hint: str = "") -> str:
    """組查核 prompt。evidence_hint 可放人工整理的公開闢謠案例摘要（MVP 先用少量範例）。"""
    quotes = "\n".join(f"- {c['content']}（可疑分數 {c.get('suspicion_score', '?')}）"
                       for c in suspicious_comments) or "（本批資料未篩出明顯可疑留言）"
    evidence_block = f"\n可比對之證據線索：\n{evidence_hint}" if evidence_hint else ""
    return (
        f"{MARK_FACT_CHECK}\n"
        "你是一位嚴謹的事實查核員。請針對下列『待查核主張』，"
        "根據提供的社群留言與證據線索進行查核，並只輸出一個 JSON 物件（不要多餘文字）。\n\n"
        f"待查核主張／主題：{topic}\n"
        f"相關可疑留言：\n{quotes}{evidence_block}\n\n"
        "JSON 需包含欄位：\n"
        "- claim：你實際查核的具體主張（字串）\n"
        f"- credibility_label：必須是 {list(CREDIBILITY_LABELS)} 其中之一\n"
        "- confidence：0~1 的信心程度（float）\n"
        "- evidence_summary：證據彙整摘要（字串，需說明依據）\n"
        "- evidence_sufficient：證據是否足以支持上述判定（true/false）\n\n"
        "重要原則：若證據不足以判定真偽，credibility_label 必須填「證據不足」，"
        "confidence 給低分，切勿武斷輸出高信心的真/偽判定。"
    )


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def apply_guardrails(fact: dict) -> dict:
    """對 LLM 回傳的 fact_check dict 做防呆與一致性修正，回傳乾淨結果。"""
    label = fact.get("credibility_label")
    if label not in CREDIBILITY_LABELS:
        # 標籤不在白名單（LLM 亂寫或格式跑掉）→ 保守處理，交人工複核
        fact["credibility_label"] = CREDIBILITY_NEEDS_REVIEW
        label = CREDIBILITY_NEEDS_REVIEW

    fact["confidence"] = _coerce_float(fact.get("confidence"), 0.0)
    fact["evidence_sufficient"] = bool(fact.get("evidence_sufficient", False))
    fact.setdefault("claim", "")
    fact.setdefault("evidence_summary", "")

    # 核心護欄：證據不足卻給了「確定性」標籤 → 降級為「證據不足」並壓低信心
    if not fact["evidence_sufficient"] and label in CONFIDENT_LABELS:
        fact["credibility_label"] = CREDIBILITY_INSUFFICIENT
        fact["confidence"] = min(fact["confidence"], INSUFFICIENT_CONFIDENCE_CAP)
    # 即使標籤本身就是「證據不足」，信心也不該過高
    if fact["credibility_label"] == CREDIBILITY_INSUFFICIENT:
        fact["confidence"] = min(fact["confidence"], INSUFFICIENT_CONFIDENCE_CAP)

    return fact


def run_fact_check(llm: LLMClient, topic: str, suspicious_comments: list,
                   evidence_hint: str = "") -> dict:
    """呼叫 LLM 做查核並套用護欄。LLM 回傳非 JSON 時，安全降級為『待人工複核』。"""
    prompt = build_fact_check_prompt(topic, suspicious_comments, evidence_hint)
    raw = llm.generate(prompt)
    try:
        fact = json.loads(raw)
        if not isinstance(fact, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        return {
            "claim": topic,
            "credibility_label": CREDIBILITY_NEEDS_REVIEW,
            "confidence": 0.0,
            "evidence_summary": "LLM 查核輸出格式無法解析，已保守標記為待人工複核。",
            "evidence_sufficient": False,
        }
    return apply_guardrails(fact)
