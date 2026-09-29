# -*- coding: utf-8 -*-
"""
真實 Claude API client（對應第十五節 LLM API；提案寫 OpenAI，本專案沿用團隊既有 Claude 實作）。
連線層重試：逾時／連線／限流 → 指數退避；金鑰／參數錯誤 → 直接拋出交上層自我診斷。
使用前：pip install anthropic；export ANTHROPIC_API_KEY=...（勿寫死或 commit）。
"""
from __future__ import annotations
import time
from .llm_client import LLMClient

DEFAULT_MODEL = "claude-sonnet-5"


class ClaudeClient(LLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, max_retries: int = 3):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def generate(self, prompt: str) -> str:
        from anthropic import APIConnectionError, APIError, APITimeoutError, RateLimitError
        last = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}])
                return "".join(b.text for b in resp.content if b.type == "text")
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
            except APIError:
                raise
        raise RuntimeError(f"Claude API 重試 {self.max_retries} 次後仍失敗：{last}")
