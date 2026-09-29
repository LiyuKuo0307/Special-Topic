# -*- coding: utf-8 -*-
"""
LLM 介接層：抽象介面 + Demo/測試用 MockLLMClient（對應第十七節 Demo Mode）。
Mock 靠 prompt 任務標記判斷要回哪種輸出：擴展／主張拆解／查核／評分／報告。
"""
from __future__ import annotations
import json
import os
import re

from .schema import CRED_HIGH_FALSE, CRED_INSUFFICIENT, EVAL_KEYS

MARK_EXPAND = "【任務：關鍵字擴展】"
MARK_DECOMPOSE = "【任務：主張拆解】"
MARK_FACT_CHECK = "【任務：假訊息查核】"
MARK_JUDGE = "【任務：報告評分 LLM-as-Judge】"
MARK_REPORT = "【任務：撰寫查核報告】"


def loads_llm_json(raw):
    """穩健解析 LLM 回傳的 JSON：去除 ```json ... ``` 圍欄、擷取第一個 {..} 或 [..] 區塊再 parse。
    解析不出來時丟 json.JSONDecodeError，交由呼叫端既有 except 保守處理。"""
    if raw is None:
        raise json.JSONDecodeError("empty", "", 0)
    s = str(raw).strip()
    m = re.match(r"^```[A-Za-z0-9]*\s*(.*?)\s*```$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for op, cl in (("{", "}"), ("[", "]")):
        start = s.find(op)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(s)):
            if s[i] == op:
                depth += 1
            elif s[i] == cl:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise json.JSONDecodeError("無法從 LLM 輸出解析出 JSON", s, 0)


class LLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Demo/測試用假 LLM，不打真實 API，產出為虛構示例。"""

    def __init__(self, fail_first_n: int = 1, always_fail: bool = False,
                 evidence_sufficient: bool = True):
        self.fail_first_n = fail_first_n
        self.always_fail = always_fail
        self.evidence_sufficient = evidence_sufficient
        self.report_calls = 0

    def generate(self, prompt: str) -> str:
        if MARK_EXPAND in prompt:
            return json.dumps(["某疫苗不孕", "疫苗後遺症", "疫苗副作用"], ensure_ascii=False)
        if MARK_DECOMPOSE in prompt:
            return json.dumps(["某疫苗會導致不孕", "衛福部未闢謠"], ensure_ascii=False)
        if MARK_FACT_CHECK in prompt:
            return self._fact_check()
        if MARK_JUDGE in prompt:
            return self._judge(prompt)
        return self._report()

    def _fact_check(self) -> str:
        if self.evidence_sufficient:
            return json.dumps({
                "claim": "某疫苗會導致不孕", "credibility_label": CRED_HIGH_FALSE,
                "confidence": 0.78,
                "evidence_summary": "衛福部與台灣事實查核中心公開資料顯示無醫學證據支持此說法（示例）。",
                "evidence_sufficient": True}, ensure_ascii=False)
        return json.dumps({
            "claim": "某疫苗會導致不孕", "credibility_label": CRED_INSUFFICIENT,
            "confidence": 0.3, "evidence_summary": "現有留言缺乏可比對之權威來源，無法判定（示例）。",
            "evidence_sufficient": False}, ensure_ascii=False)

    def _judge(self, prompt: str) -> str:
        specific = ("闢謠" in prompt or "衛福部" in prompt) and "證據" in prompt
        s = 4 if specific else 2
        return json.dumps({"specificity": s, "readability": 4, "relevance": 4 if specific else 3,
                           "risk_reasoning": s,
                           "feedback": "已補充具體證據與來源" if specific else "報告缺乏具體證據引用"},
                          ensure_ascii=False)

    def _report(self) -> str:
        self.report_calls += 1
        if self.always_fail or self.report_calls <= self.fail_first_n:
            return "整體留言對這個說法有不少疑慮，真假難辨，需要再觀察。"
        return ("本次查核顯示社群留言約半數提及該說法之疑慮；經與衛福部及台灣事實查核中心公開闢謠資料"
                "比對，查無醫學證據支持「某疫苗會導致不孕」，判定為高度疑似不實。代表留言如"
                "「這是舊謠言了衛福部很早就闢謠過」。本結論證據充足（示例，Demo Mode 產出）。")


class CachedLLMClient(LLMClient):
    """LLM 快取包裝（第十七節「快取機制」）：相同 prompt 只呼叫底層一次，
    降低重複請求對外部資源之依賴（重試、重複查核時特別有用），也讓 Demo 更穩定。
    """

    def __init__(self, inner: "LLMClient", cache: dict | None = None,
                 cache_path: str | None = None):
        self.inner = inner
        self.cache_path = cache_path
        if cache is None and cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
        self._cache = cache if cache is not None else {}
        self.hits = 0
        self.misses = 0

    def _save(self):
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)
        except Exception:
            pass

    def generate(self, prompt: str) -> str:
        if prompt in self._cache:
            self.hits += 1
            return self._cache[prompt]
        self.misses += 1
        out = self.inner.generate(prompt)
        self._cache[prompt] = out
        self._save()   # 跨執行沿用（第十七節本地快取）
        return out

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "cached_prompts": len(self._cache)}


class SafeLLMClient(LLMClient):
    """把底層 LLM 的例外（限流 429／斷線／型號全失敗等）吞掉，回傳空字串讓下游保守降級，
    避免單次 API 失敗直接讓整條查核崩潰。記錄最後一次錯誤供診斷（self.last_error）。"""

    def __init__(self, inner: "LLMClient"):
        self.inner = inner
        self.last_error = None

    def generate(self, prompt: str) -> str:
        try:
            return self.inner.generate(prompt)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            from .debug_log import log_event
            log_event("llm_exception", error=self.last_error, prompt_head=str(prompt)[:120])
            return ""


def validate_judge_json(raw: str) -> dict:
    try:
        s = loads_llm_json(raw)
        if not isinstance(s, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        s = {k: 1 for k in EVAL_KEYS}
        s["feedback"] = "評分格式解析失敗，視為未達標"
    return s


def make_llm(api_key: str | None = None, provider: str = "auto"):
    """建立 LLM client，回傳 (llm, is_demo)。
    優先順序（provider='auto'）：Gemini（免費）→ Claude → Demo(Mock)。
    - api_key：使用者手動提供的金鑰（視為 Gemini 金鑰，優先於環境變數）。
    - provider：可強制 'gemini' / 'claude' / 'mock'。
    真實 client 皆以 CachedLLMClient 包裝（第十七節快取）。
    """
    gem_env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    ant_env = os.environ.get("ANTHROPIC_API_KEY")

    if provider == "mock":
        return MockLLMClient(fail_first_n=1), True

    # Gemini 優先（手動金鑰視為 Gemini；或環境有 Gemini 金鑰）
    if provider == "gemini" or (provider == "auto" and (api_key or gem_env)):
        key = api_key or gem_env
        if key:
            try:
                from .gemini_client import GeminiClient
                return SafeLLMClient(CachedLLMClient(GeminiClient(api_key=key))), False
            except Exception:
                pass  # Gemini 起不來就往下嘗試

    # Claude 次之（環境有 Anthropic 金鑰，或強制指定）
    if provider == "claude" or (provider == "auto" and ant_env):
        try:
            from .claude_client import ClaudeClient
            return SafeLLMClient(CachedLLMClient(ClaudeClient(api_key=ant_env))), False
        except Exception:
            pass

    return MockLLMClient(fail_first_n=1), True
