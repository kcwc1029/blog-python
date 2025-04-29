# 📂 Project: 簡易版 RAG 系統 - PDF 文件讀取與關聯問答

![upgit_20250418_1744906064.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250418_1744906064.png)

## 🔍 作用簡介

這個專案是一個簡易專案，執行 **PDF 文件讀取 -> 分段 -> 編碼記憶 (Embeddings) -> 儲存 ChromaDB -> 讓 LLM 來對答問答 (RAG)** 的流程，全程操作都會用簡單的 **Streamlit 界面**進行！

## 🔧 執行流程

1. **PDF 文件上傳**：使用 Streamlit 上傳一或多個 PDF
2. **PDF 讀取 & 分段處理**

    - 將文字按正文關鍵處分節，避免截斷中文句子
    - 每段大約 1000 字，並有重疊 (Overlap)

3. **將分段後的內容轉成 Embeddings**：支援 OpenAI / Ollama 等多種型的編碼模型

4. **把文件分段與編碼記憶儲存到 ChromaDB**：使用 Persistent Client，可持續儲存文件資料

5. **問題搜尋 (RAG)**

    - 用户在網頁上輸入問題
    - 從已儲存的分段中，找最相關的段落

6. 給大型語言模型將 Context + 問題一起給生成答案
7. 顯示回答與來源段落

## 📄 檔案組織

```
└── Project
    ├── main.py                # Streamlit 啟動程式
    ├── rag
    │    ├── model_selector.py  # 模型選擇器
    │    ├── pdf_processor.py  # PDF 分段處理器
    │    └── rag_system.py      # RAG 系統主體
    └── requirements.txt      # 監認安裝的套件列表
```

## 🌐 啟動步驟

1. **安裝監認套件**

```bash
pip install -r requirements.txt
```

2. **執行 Streamlit App**

```bash
streamlit run main.py
```

## 🔎 功能內容

| 功能          | 說明                                 |
| :------------ | :----------------------------------- |
| 文件上傳      | 支援上傳多個 PDF                     |
| 分段處理      | 根據字數分割，並保留一定的關聯距離   |
| ChromaDB 儲存 | 將分段與編碼記憶存入，有持續性       |
| 自由問題搜尋  | 能輸入問題，給予聯繫答案             |
| 來源段落顯示  | 回答不介意，可以繳出最關聯的原文段落 |
