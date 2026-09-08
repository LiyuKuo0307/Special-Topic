# -*- coding: utf-8 -*-
"""
真正呼叫 Anthropic Claude API 的 LLMClient 實作（對應企劃書第六節「查核與可信度評估：Claude API」）。

使用前：
  1. pip install anthropic
  2. 設定環境變數：export ANTHROPIC_API_KEY="你的金鑰"
     不要把金鑰寫死在程式碼裡、也不要 commit 進 GitHub。

對應企劃書第十五節「自我評估迴圈失控」以外的另一層——API 連線層重試：
  逾時／連線錯誤／限流 → 指數退避重試；金鑰或參數錯誤 → 直接往外拋（重試沒用）。
這一層只管「API 有沒有正常回應」，跟 agent_optimization_loop.py 的「內容品質好不好」是兩件事。
"""

from __future__ import annotations

import time

from llm_client import LLMClient

# 之後 Anthropic 更新模型名稱時，請查 https://docs.claude.com/en/docs/about-claude/models 確認最新代稱。
DEFAULT_MODEL = "claude-sonnet-5"


class ClaudeClient(LLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, max_retries: int = 3):
        # 延遲匯入，讓沒安裝 anthropic 的環境仍能 import 專案其他模組（Demo Mode 不需要這個套件）。
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)  # api_key=None 時 SDK 自動讀環境變數
        self.model = model
        self.max_retries = max_retries

    def generate(self, prompt: str) -> str:
        from anthropic import (
            APIConnectionError,
            APIError,
            APITimeoutError,
            RateLimitError,
        )

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join(block.text for block in resp.content if block.type == "text")
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_err = e  # 網路抖動／逾時／限流：重試通常有用，指數退避
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
            except APIError:
                raise  # 金鑰／參數錯誤：重試沒用，往外拋給上層自我診斷處理

        raise RuntimeError(f"Claude API 重試 {self.max_retries} 次後仍失敗：{last_err}")


if __name__ == "__main__":
    from anthropic import APIError
    print("用無效金鑰測試錯誤處理（預期直接拋例外，不應卡住或誤重試）...")
    client = ClaudeClient(api_key="invalid-key-for-testing")
    try:
        client.generate("test")
        print("❌ 預期應該失敗，但沒有拋出例外")
    except APIError as e:
        print(f"✅ 正確拋出 APIError，沒有進入無謂重試：{type(e).__name__}")
    except Exception as e:
        print(f"⚠️ 拋出非預期例外 {type(e).__name__}：{e}")
