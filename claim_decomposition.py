# -*- coding: utf-8 -*-
"""
主張拆解 claim decomposition（對應第十一節：由 LLM 將複合陳述拆解為可個別查證之具體主張）。
"""
from __future__ import annotations
import json

from .llm_client import LLMClient, MARK_DECOMPOSE, loads_llm_json


def build_decompose_prompt(topic: str, suspicious_comments: list) -> str:
    quotes = "；".join(c["content"] for c in suspicious_comments[:5]) or topic
    return (f"{MARK_DECOMPOSE}\n請把下列與主題相關的社群內容，拆解成可個別查證的具體主張（claim），"
            "每則為一句可判定真偽的陳述，只回 JSON 陣列。不得把含糊句型擅自收斂成單一解釋；"
            "若『來自／起源』可能混淆最早發現地、命名地點與成因或來源，必須拆成不同主張，"
            f"而且不得把地名直接當作起源證據。\n主題：{topic}\n內容：{quotes}")


def decompose_claims(llm: LLMClient, topic: str, suspicious_comments: list) -> list:
    try:
        arr = loads_llm_json(llm.generate(build_decompose_prompt(topic, suspicious_comments)))
        if isinstance(arr, list) and arr:
            return [str(x) for x in arr]
    except Exception:
        pass
    return [topic]  # 拆解失敗時退回原主題作為單一主張
