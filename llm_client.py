# -*- coding: utf-8 -*-
"""
LLM 介接層：抽象介面 LLMClient + 測試/Demo 用的 MockLLMClient。

對應企劃書多處：
- 第六節技術選型「查核與可信度評估：Claude API」「Agent 自我評估：LLM-as-Judge」
- 修訂說明「明確區分 Demo Mode（MockLLMClient，產出報告文字為虛構示例）與真實 MVP」

設計原則：所有 LLM 呼叫都走 LLMClient.generate() 這一個介面。
- Demo Mode / 測試：用 MockLLMClient（不打真實 API，回傳可預期的假資料，用來驗證「迴圈邏輯」）
- 真實 MVP：用 claude_client.ClaudeClient（實作同一介面）
換 client 時 pipeline / agent loop 一行都不用改。

MockLLMClient 靠 prompt 裡的「任務標記」字串判斷這次要回哪一種輸出：
  MARK_FACT_CHECK → 回 fact_check JSON
  MARK_JUDGE      → 回 LLM-as-Judge 評分 JSON
  其他             → 回查核報告文字
這幾個標記由 fact_check.py / agent_optimization_loop.py 在組 prompt 時嵌入。
"""

from __future__ import annotations

import json

from schema import (
    CREDIBILITY_LIKELY_FALSE,
    CREDIBILITY_INSUFFICIENT,
    EVAL_KEYS,
)

# prompt 任務標記（同時給真實 LLM 當指示、給 Mock 當分派依據）
MARK_FACT_CHECK = "【任務：假訊息查核】"
MARK_JUDGE = "【任務：報告評分 LLM-as-Judge】"
MARK_REPORT = "【任務：撰寫查核報告】"


class LLMClient:
    """抽象介面。接真實 API 時寫一個新 class 實作 generate() 即可，其餘程式碼不用改。"""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """
    Demo Mode / 測試用假 LLM，不打真實 API。產出為「虛構示例」，非真實查核結果。

    參數：
    - fail_first_n：前 n 次「報告生成」故意產出空泛版本，用來驗證自我優化迴圈真的會重生。
    - always_fail：報告永遠空泛，用來驗證「重試用盡 → 標記人工複核」路徑。
    - evidence_sufficient：控制假 fact_check 是否「證據充足」，方便測有／無足夠證據兩條路。
    """

    def __init__(self, fail_first_n: int = 1, always_fail: bool = False,
                 evidence_sufficient: bool = True):
        self.fail_first_n = fail_first_n
        self.always_fail = always_fail
        self.evidence_sufficient = evidence_sufficient
        self.report_calls = 0

    def generate(self, prompt: str) -> str:
        if MARK_FACT_CHECK in prompt:
            return self._fact_check(prompt)
        if MARK_JUDGE in prompt:
            return self._judge(prompt)
        return self._report(prompt)

    # --- 假 fact_check 輸出 ---
    def _fact_check(self, prompt: str) -> str:
        if self.evidence_sufficient:
            return json.dumps({
                "claim": "某疫苗會導致不孕",
                "credibility_label": CREDIBILITY_LIKELY_FALSE,
                "confidence": 0.78,
                "evidence_summary": "衛福部公開闢謠資料顯示無醫學證據支持此說法（示例，Demo Mode 產出）。",
                "evidence_sufficient": True,
            }, ensure_ascii=False)
        # 證據不足時：不得輸出高信心的真/偽判定（由 fact_check.py 的護欄再保證一次）
        return json.dumps({
            "claim": "某疫苗會導致不孕",
            "credibility_label": CREDIBILITY_INSUFFICIENT,
            "confidence": 0.3,
            "evidence_summary": "目前提供的留言中缺乏可比對之權威來源，無法判定真偽（示例）。",
            "evidence_sufficient": False,
        }, ensure_ascii=False)

    # --- 假 LLM-as-Judge 評分輸出 ---
    def _judge(self, prompt: str) -> str:
        # 報告中同時出現具體留言與證據字樣才給高分，用來檢驗優化迴圈確實提升了具體性。
        specific = ("闢謠" in prompt or "衛福部" in prompt) and "證據" in prompt
        score = 4 if specific else 2
        return json.dumps({
            "specificity": score,
            "readability": 4,
            "relevance": 4 if specific else 3,
            "risk_reasoning": score,
            "feedback": "已補充具體證據與來源" if specific else "報告缺乏具體證據引用，可信度判斷理由不足",
        }, ensure_ascii=False)

    # --- 假查核報告輸出 ---
    def _report(self, prompt: str) -> str:
        self.report_calls += 1
        vague = self.always_fail or (self.report_calls <= self.fail_first_n)
        if vague:
            return "整體留言對這個說法有不少疑慮，真假難辨，需要再觀察。"
        return (
            "本次查核顯示，社群留言中約半數提及該說法之疑慮；經與衛福部公開闢謠資料比對，"
            "查無醫學證據支持「某疫苗會導致不孕」之主張，故判定為高度疑似不實訊息。"
            "代表留言如「這是舊謠言了衛福部很早就闢謠過」。"
            "本結論證據充足，信心程度中高（示例，Demo Mode 產出，非真實查核結果）。"
        )


def _validate_judge_json(raw: str) -> dict:
    """把 LLM-as-Judge 回傳的字串解析成分數 dict；解析失敗時回全 1 分（視為未達標）。"""
    try:
        scores = json.loads(raw)
        if not isinstance(scores, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        scores = {k: 1 for k in EVAL_KEYS}
        scores["feedback"] = "評分格式解析失敗，視為未達標"
    return scores
