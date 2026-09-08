import os
import google.generativeai as genai
from llm_client import LLMClient

# 預設使用的 Gemini 模型名稱
DEFAULT_MODEL = "gemini-3.6-flash"

class GeminiClient(LLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        # 設定 API 金鑰
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # 如果沒傳入，會自動抓環境變數 GEMINI_API_KEY
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            
        self.model_name = model

    def generate(self, prompt: str) -> str:
        try:
            # 使用舊版穩定 SDK 產生內容
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API 呼叫失敗：{e}")