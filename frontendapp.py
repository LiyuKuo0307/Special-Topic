import streamlit as st
import pandas as pd

st.title("專題資料上傳與預覽")

# 檔案上傳區
uploaded_file = st.file_uploader("請選擇或拖曳 CSV 檔案：", type=["csv"])

# 資料預覽區
st.subheader("資料預覽：")

if uploaded_file is not None:
    # 讀取 CSV (第一列會自動被當作欄位標題/檔名相關屬性)
    df = pd.read_csv(uploaded_file)
    st.dataframe(df) # 自動產生精美的互動式表格
    
    # 預留按鈕：點擊後呼叫整合者的後端
    if st.button("開始進行 AI 分析與查核"):
        st.info("分析中，請稍候...")
        # 這裡未來直接呼叫你的後端處理函式 (例如 process_entire_flow(df))
else:
    st.info("尚未上傳檔案")