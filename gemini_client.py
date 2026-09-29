# -*- coding: utf-8 -*-
"""
真實 Gemini API client（Google Gemini，免費層可用）。
介面與 ClaudeClient 相同：generate(prompt) -> str，可與 CachedLLMClient、pipeline 無縫替換。

使用前：
    pip install google-genai          # 新版 SDK（建議）；或 pip install google-generativeai（舊版亦支援）
    金鑰申請：https://aistudio.google.com/apikey  （免費、免綁卡）
    設定金鑰（勿寫死或 commit）：
        Windows cmd :  set GEMINI_API_KEY=你的金鑰
        或直接在 Dashboard 側邊欄貼上

型號：Google 會不定期汰換型號。本 client 具「自我修復」：
  1) 依 GEMINI_MODEL 環境變數或預設候選清單嘗試；
  2) 若 API 回「請改用 models/XXX」，會自動抓出建議型號並改用它重試。
連線層重試：逾時／連線／限流 → 指數退避。
"""
from __future__ import annotations
import os
import re
import time
from llm_client import LLMClient

# 免費層候選型號（依序嘗試；可用 GEMINI_MODEL 覆蓋）。就算全過期，也會用 API 建議的型號自我修復。
_CANDIDATE_MODELS = [
    os.environ.get("GEMINI_MODEL", "").strip() or "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-flash",
]

_UNAVAILABLE_HINTS = ("not found", "not supported", "404", "no longer available",
                      "unknown model", "invalid model", "unavailable")


class GeminiClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 max_retries: int = 3, use_search: bool = False):  # 接地需付費層，預設關閉；有帳單可設 True
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("缺少 Gemini 金鑰：請設定 GEMINI_API_KEY 或在建立時傳入 api_key。")
        pref = [m for m in ([model] if model else []) + _CANDIDATE_MODELS if m]
        seen = set()
        self._models = [m for m in pref if not (m in seen or seen.add(m))]
        self.max_retries = max_retries
        self.use_search = use_search   # 開啟 Google 搜尋接地（讓 LLM 有依據可查證）
        self._search_warned = False
        self._backend = None
        self._client = None
        self._init_backend()

    def _init_backend(self):
        """優先用新版 SDK（google-genai），失敗則退回舊版（google-generativeai）。"""
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._backend = "new"
            return
        except Exception:
            pass
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai_old = genai
            self._backend = "old"
            return
        except Exception as e:
            raise RuntimeError(
                "無法載入 Gemini SDK，請先安裝：pip install google-genai（或 google-generativeai）。"
                f" 原始錯誤：{e}")

    def _search_config(self):
        """建立含 Google 搜尋接地的設定；失敗回 None（例如 SDK 版本不支援）。"""
        try:
            from google.genai import types
            return types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())])
        except Exception:
            return None

    def _call_once(self, model: str, prompt: str) -> str:
        if self._backend == "new":
            cfg = self._search_config() if self.use_search else None
            if cfg is not None:
                try:
                    resp = self._client.models.generate_content(
                        model=model, contents=prompt, config=cfg)
                    return (getattr(resp, "text", "") or "").strip()
                except Exception as e:
                    # 接地失敗（如免費層未開通/不支援）→ 記一次、之後整個關閉接地，避免每筆多打一次浪費額度
                    if not self._search_warned:
                        from .debug_log import log_event
                        log_event("grounding_unavailable", error=f"{type(e).__name__}: {e}")
                        self._search_warned = True
                    self.use_search = False
            resp = self._client.models.generate_content(model=model, contents=prompt)
            return (getattr(resp, "text", "") or "").strip()
        gm = self._genai_old.GenerativeModel(model)
        resp = gm.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip()

    @staticmethod
    def _suggested_model(err_msg: str) -> str | None:
        """從錯誤訊息抓 API 建議的替代型號，例如『use models/gemini-3.5-flash-lite』。"""
        m = re.search(r"use\s+models/([A-Za-z0-9.\-]+)", err_msg)
        if m:
            return m.group(1)
        # 退而求其次：抓訊息裡最後一個 models/XXX
        allm = re.findall(r"models/([A-Za-z0-9.\-]+)", err_msg)
        return allm[-1] if allm else None

    def generate(self, prompt: str) -> str:
        tried: set = set()
        queue = list(self._models)
        last_err = None
        while queue:
            model = queue.pop(0)
            if model in tried:
                continue
            tried.add(model)
            for attempt in range(1, self.max_retries + 1):
                try:
                    text = self._call_once(model, prompt)
                    if text:
                        # 記住成功型號，之後同一個 client 直接優先用它
                        self._models = [model] + [m for m in self._models if m != model]
                        return text
                    last_err = RuntimeError("Gemini 回傳空內容")
                except Exception as e:
                    last_err = e
                    msg = str(e)
                    low = msg.lower()
                    # API 建議的替代型號 → 自動加入嘗試佇列（自我修復）
                    sug = self._suggested_model(msg)
                    if sug and sug not in tried and sug not in queue:
                        queue.append(sug)
                    # 型號不可用 → 直接換下一個，不重試此型號
                    if any(k in low for k in _UNAVAILABLE_HINTS):
                        break
                    # 其他（限流/逾時/連線）→ 指數退避後重試
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini API 呼叫失敗（已試型號 {sorted(tried)}）：{last_err}")
