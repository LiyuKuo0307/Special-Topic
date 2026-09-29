# -*- coding: utf-8 -*-
"""
五級可信度評估（對應第十一節）：主張拆解 → 證據蒐集 → LLM 查核 → 五級標籤 + 護欄。
五級：高度可信／可能可信／證據不足／可能不實／高度疑似不實（第十一節一）。
護欄（第十一/十八節）：證據不足不得給高信心確定性標籤，避免武斷判定與名譽風險。
"""
from __future__ import annotations
import json

from .llm_client import LLMClient, MARK_FACT_CHECK, loads_llm_json
from .schema import (ALL_LABELS, CREDIBILITY_LABELS, CRED_INSUFFICIENT,
                     CRED_NEEDS_REVIEW, CONFIDENT_LABELS)

INSUFFICIENT_CONFIDENCE_CAP = 0.4

_AUTHORITY_PHRASES = (
    "全球公共衛生機構", "世界衛生組織", "WHO", "世衛", "疾管署", "CDC", "衛福部",
    "司法院", "檢察署", "科學界的普遍認知", "科學界普遍認知", "公認的事實",
)


def build_fact_check_prompt(topic: str, claims: list, suspicious: list, evidence_hint: str = "") -> str:
    claim_block = "\n".join(f"- {c}" for c in claims) or f"- {topic}"
    quotes = "\n".join(f"- {c['content']}" for c in suspicious[:5]) or "（無明顯可疑留言）"
    ev = f"\n可比對之證據線索：\n{evidence_hint}" if evidence_hint else "\n（無外部證據線索；若主張屬公認的錯誤或正確常識，可依你的既有知識判定，不必因此回證據不足）"
    return (f"{MARK_FACT_CHECK}\n你是嚴謹的事實查核員，請針對下列主張查核，只輸出一個 JSON 物件。\n"
            f"主題：{topic}\n待查核主張：\n{claim_block}\n相關留言：\n{quotes}{ev}\n\n"
            f"JSON 欄位：claim、credibility_label（必須是 {list(CREDIBILITY_LABELS)} 之一）、"
            "confidence（0~1）、evidence_summary、evidence_sufficient（true/false）。\n"
            "判定原則（依證據與公認事實知識判定，勿保守過頭、也勿武斷）：\n"
            "1. 這裡的『證據』包含兩類：(a) 上方外部證據線索；(b) 廣為確立的科學或事實常識。只要其一足以判斷，evidence_sufficient 就設 true。\n"
            "2. 若主張明顯違反公認科學/事實（例：疫苗使人不孕、5G 傳播病毒、喝酒精殺體內病毒、某偏方治百病），"
            "請依既有知識判為對應不實標籤（可能不實／高度疑似不實）、evidence_sufficient=true、給合理 confidence，不要回「證據不足」。\n"
            "3. 若主張與公認事實相符或屬中性事實陳述，判「可能可信／高度可信」；對個人意見、情緒抒發、無法證真偽的價值判斷，不要當假訊息冤枉。\n"
            "4. 但不得虛構『具體可查證事實』：上方未提供的特定影片/貼文內容、日期、確診數字、司法進度、官方文件細節等，無證據就不得編造，"
            "這類具體事件缺證據時回「證據不足」；evidence_summary 需說明依據的是公認常識或上方證據。\n"
            f"5. 最終 claim 必須完全聚焦並填寫為主題「{topic}」；拆解主張只是輔助，"
            "不得把其中一個旁支內容改成最終查核標的。\n"
            "6. 若主題含『來自／起源』，必須區分最早發現或通報地點與真正來源／成因；"
            "沒有直接證據時不得把兩者視為同一件事。")


def _coerce_float(v, default=0.0):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def apply_guardrails(fact: dict) -> dict:
    label = fact.get("credibility_label")
    if label not in ALL_LABELS:
        fact["credibility_label"] = CRED_NEEDS_REVIEW
        label = CRED_NEEDS_REVIEW
    fact["confidence"] = _coerce_float(fact.get("confidence"))
    fact["evidence_sufficient"] = bool(fact.get("evidence_sufficient", False))
    fact.setdefault("claim", "")
    fact.setdefault("evidence_summary", "")
    if not fact["evidence_sufficient"] and label in CONFIDENT_LABELS:
        fact["credibility_label"] = CRED_INSUFFICIENT
        fact["confidence"] = min(fact["confidence"], INSUFFICIENT_CONFIDENCE_CAP)
    if fact["credibility_label"] in (CRED_INSUFFICIENT, CRED_NEEDS_REVIEW):
        fact["confidence"] = min(fact["confidence"], INSUFFICIENT_CONFIDENCE_CAP)
    return fact


def apply_evidence_grounding_guardrail(fact: dict, evidence_hint: str) -> dict:
    """禁止模型用未出現在輸入來源中的機構、共識或『公認事實』補足證據。"""
    fact = apply_guardrails(fact)
    if not fact.get("evidence_sufficient"):
        return fact
    summary = str(fact.get("evidence_summary") or "")
    hint = str(evidence_hint or "")
    unsupported = [phrase for phrase in _AUTHORITY_PHRASES if phrase in summary and phrase not in hint]
    if unsupported:
        fact["credibility_label"] = CRED_INSUFFICIENT
        fact["confidence"] = min(float(fact.get("confidence", 0) or 0), 0.2)
        fact["evidence_sufficient"] = False
        fact["evidence_summary"] = (
            "模型輸出引用了未出現在本次可追溯來源中的機構、共識或公認事實（"
            + "、".join(unsupported)
            + "），不能作為本次判定依據，故保守標記為證據不足。"
        )
    return fact


def _one_fact_check(llm, prompt, topic, evidence_hint):
    """單次查核：呼叫 LLM → 穩健解析 → 護欄。失敗回 None（不崩）。"""
    from .debug_log import log_event
    raw = ""
    try:
        raw = llm.generate(prompt)
        fact = loads_llm_json(raw)
        if not isinstance(fact, dict):
            raise ValueError
        fact = apply_evidence_grounding_guardrail(fact, evidence_hint)
        fact["claim"] = topic
        return fact
    except Exception as e:
        log_event("fact_check_fail", topic=topic, error=f"{type(e).__name__}: {e}", raw=str(raw))
        return None


def run_fact_check(llm: LLMClient, topic: str, claims: list, suspicious: list,
                   evidence_hint: str = "", samples: int = 1) -> dict:
    """五級可信度查核。samples>1 時做「多次取多數」(self-consistency)，降低 LLM 隨機性；
    多次結果平手（模型分歧）→ 保守回『證據不足』。"""
    from collections import Counter
    from .debug_log import log_event
    prompt = build_fact_check_prompt(topic, claims, suspicious, evidence_hint)
    n = max(1, int(samples or 1))
    facts = [
        f for f in (
            _one_fact_check(llm, prompt, topic, evidence_hint) for _ in range(n)
        ) if f
    ]

    if not facts:
        return {"claim": topic, "credibility_label": CRED_NEEDS_REVIEW, "confidence": 0.0,
                "evidence_summary": "LLM 查核呼叫失敗（可能限流/斷線）或輸出無法解析，保守標記待人工複核。",
                "evidence_sufficient": False}

    if n == 1:
        f = facts[0]
        log_event("fact_check_ok", topic=topic, label=f["credibility_label"], confidence=f["confidence"])
        return f

    counts = Counter(f["credibility_label"] for f in facts)
    top_label, top_n = counts.most_common(1)[0]
    tie = sum(1 for _, c in counts.items() if c == top_n) > 1
    if tie:
        result = {"claim": topic, "credibility_label": CRED_INSUFFICIENT, "confidence": 0.3,
                  "evidence_summary": f"多次查核結果不一致（{dict(counts)}），模型判斷分歧，保守標記證據不足。",
                  "evidence_sufficient": False}
    else:
        majority = [f for f in facts if f["credibility_label"] == top_label]
        result = dict(majority[0])
        result["confidence"] = round(sum(f["confidence"] for f in majority) / len(majority), 2)
    result = apply_guardrails(result)
    result["samples"] = n
    result["label_votes"] = dict(counts)
    log_event("fact_check_vote", topic=topic, votes=dict(counts), final=result["credibility_label"])
    return result
