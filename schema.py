# -*- coding: utf-8 -*-
"""
共用結構與常數（對應正式提案 v2 第十一節五級可信度、第十節情緒、第十六節評估）。
集中跨模組共用字串常數，避免各檔案寫死不同字面值造成整合對不上。
"""
from __future__ import annotations

# ---------- 情緒標籤（第十節） ----------
SENTIMENT_POSITIVE = "positive"
SENTIMENT_NEUTRAL = "neutral"
SENTIMENT_NEGATIVE = "negative"
SENTIMENTS = (SENTIMENT_POSITIVE, SENTIMENT_NEUTRAL, SENTIMENT_NEGATIVE)
SENTIMENT_ZH = {SENTIMENT_POSITIVE: "正面", SENTIMENT_NEUTRAL: "中立", SENTIMENT_NEGATIVE: "負面"}

# ---------- 五級可信度標籤（第十一節，含「證據不足」） ----------
CRED_HIGH_TRUE = "高度可信"
CRED_MAYBE_TRUE = "可能可信"
CRED_INSUFFICIENT = "證據不足"
CRED_MAYBE_FALSE = "可能不實"
CRED_HIGH_FALSE = "高度疑似不實"
CRED_NEEDS_REVIEW = "待人工複核"  # LLM 輸出異常時的保守降級（非五級之一，但需存在）

CREDIBILITY_LABELS = (CRED_HIGH_TRUE, CRED_MAYBE_TRUE, CRED_INSUFFICIENT,
                      CRED_MAYBE_FALSE, CRED_HIGH_FALSE)
ALL_LABELS = CREDIBILITY_LABELS + (CRED_NEEDS_REVIEW,)

# 這些代表「已有足夠證據做出偏向真/偽的判定」，不足時不得搭配高信心
CONFIDENT_LABELS = (CRED_HIGH_TRUE, CRED_MAYBE_TRUE, CRED_MAYBE_FALSE, CRED_HIGH_FALSE)
# 需要在介面上提示「參考官方查核」的高風險判定（第十一/十八節）
HIGH_RISK_LABELS = (CRED_MAYBE_FALSE, CRED_HIGH_FALSE)

# ---------- LLM-as-Judge 自我評估指標（第七節） ----------
EVAL_KEYS = ("specificity", "readability", "relevance", "risk_reasoning")
EVAL_ZH = {"specificity": "具體性", "readability": "可讀性",
           "relevance": "對應性", "risk_reasoning": "查核判斷合理性"}


def empty_result(topic: str) -> dict:
    """欄位齊全的空結果骨架，確保下游拿到的鍵一致。"""
    return {
        "topic": topic, "expanded_keywords": [], "total_count": 0,
        "sentiment": {s: 0 for s in SENTIMENTS},
        "sentiment_ratio": {s: 0.0 for s in SENTIMENTS},
        "keywords": [], "daily_trend": [], "suspicious_comments": [],
        "representative_comments": [], "claims": [], "fact_check": None,
        "propagation_events": [], "author_graph": None, "agent_self_evaluation": None,
        "ai_report": None, "agent_log": {},
    }
