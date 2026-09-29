# -*- coding: utf-8 -*-
"""
動態關鍵字／事件擴展（對應第六節 Agent 決策、第二十一節創新點 Dynamic query expansion）。
假訊息常有變體用詞，Agent 依主題自動擴展搜尋關鍵字。
規則式 + 可選 LLM 擴展（有 LLM 時產生同義／變體詞）。
"""
from __future__ import annotations
import json

# 常見主題的別稱／變體（規則式基礎詞庫，之後可持續擴充）
_VARIANTS = {
    "疫苗": ["疫苗", "打針", "接種", "副作用", "不孕", "後遺症"],
    "食安": ["食安", "食品安全", "毒", "添加物", "致癌"],
    "地震": ["地震", "強震", "餘震", "預言", "國家級警報"],
}


def rule_based_expand(topic: str) -> list:
    topic = str(topic or "")   # 容錯：topic 非字串/None（第十七節）
    kws = {topic.strip()}
    for key, variants in _VARIANTS.items():
        if key in topic:
            kws.update(variants)
    return sorted(kws)


def llm_expand(llm, topic: str, max_terms: int = 8) -> list:
    """用 LLM 產生變體關鍵字；失敗則退回規則式。llm 為實作 generate() 的物件。"""
    from .llm_client import MARK_EXPAND, loads_llm_json
    prompt = (f"{MARK_EXPAND}\n針對查核主題「{topic}」，列出最多 {max_terms} 個"
              "可能的變體說法、別稱或相關關鍵字，只回 JSON 陣列。")
    try:
        arr = loads_llm_json(llm.generate(prompt))
        if isinstance(arr, list) and arr:
            return sorted({str(topic or "").strip(), *[str(x) for x in arr]})
    except Exception:
        pass
    return rule_based_expand(topic)


def expand_keywords(topic: str, llm=None, use_llm: bool = False) -> list:
    return llm_expand(llm, topic) if (use_llm and llm) else rule_based_expand(topic)
