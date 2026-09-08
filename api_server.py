import sys
import os
from dotenv import load_dotenv
# 把當前資料夾強制加入 Python 的模組搜尋路徑中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from fastapi import FastAPI
from fact_check import run_fact_check
from gemini_client import GeminiClient  # 改成引入 Gemini
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 1. 引入 CORS 中介軟體
from fact_check import run_fact_check
from gemini_client import GeminiClient

load_dotenv()
API_KEY = os.getenv("API_KEY")

app = FastAPI()

# 2. 加入這段 CORS 設定，允許所有來源進行測試
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 GeminiClient

# 或是透過環境變數設定 GEMINI_API_KEY
try:
    llm_client = GeminiClient(api_key=API_KEY)
except Exception as e:
    print(f"⚠️ 初始化 GeminiClient 失敗: {e}")
    llm_client = None

@app.post("/check")
def handle_check(data: dict):
    topic = data.get("topic", "")
    comments = data.get("suspicious_comments", [])
    
    if llm_client is None:
        return {"error": "LLM Client 未成功初始化，請檢查環境變數 GEMINI_API_KEY 是否設定正確。"}
    
    # 呼叫查核邏輯
    result = run_fact_check(llm_client, topic, comments)
    
    return result