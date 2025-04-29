定義：LLM 是一種使用神經網路（特別是 Transformer 架構）訓練的大型模型，透過大量語言資料學習語言結構與語意，能夠自動產生「人類語言風格的內容」。

功能特色：
- 透過「預測下一個詞」完成句子生成
- 可以進行問答、翻譯、摘要、寫作等任務
- 理解輸入內容是透過統計模式而非語意本身

## 1. 訓練流程簡述（LLM Training Pipeline）

|步驟|說明|
|---|---|
|Data Collection|收集龐大的文字語料（文章、網頁、新聞、書籍等）|
|Tokenization|將文字分割為 model 可理解的最小單位（tokens）|
|Training|使用神經網路學習輸入-輸出對應關係|
|Fine-tuning|根據特定任務進行微調（如醫療、程式碼、客服模型）|

## 2. 關鍵結構：Transformer 架構
- **Encoder-Decoder 架構**：雙向理解 + 單向生成
- **Self-Attention（自注意力）**：模型能聚焦在輸入中「重要」的字
- **Positional Encoding**：補充序列資訊
    
### 2.1. Encoder-Decoder

| 模組     | 功能                         | 類比                         |
|----------|------------------------------|------------------------------|
| Encoder  | 理解輸入句子內容               | 中文老師閱讀一篇英文文章         |
| Decoder  | 根據理解生成新句子（輸出）       | 中文老師把這篇文章翻譯成中文       |

Encoder 把輸入句子（例如英文）轉成數學向量（意思向量）
Decoder 再根據這個向量，一個字一個字地產生輸出句子（例如中文）

### 2.2. **Self-Attention（自注意力機制）**
功能： 模型在處理每一個詞時，都會檢查這個詞跟句中其他詞的關係，決定「要注意誰」。
「這句話裡，哪些字最重要？」
類比：  像你看一句話時，會特別注意關鍵字，例如在「我昨天吃了拉麵」中，"吃" 和 "拉麵" 是最重要的資訊。
例句：「The cat sat on the mat」處理 "cat" 時，模型會發現 "sat" 比 "the" 更有關係 → 給它更高注意力分數

### 2.3. ✅ 3. **Positional Encoding（位置編碼）**

因為 Transformer 是同時處理整段句子（不像 RNN 是一個字一個字走），所以它不知道哪個字是第幾個。
解法：每個詞加上一組「位置向量」來標示它在句子裡的順序。
例子：「我 很 喜歡 你」 vs 「你 喜歡 很 我」 => 用位置編碼，模型才知道「誰先誰後」影響語意
類比：  就像一組積木中每塊都有編號（順序），你才知道怎麼拼對。
> Transformer 沒有時間概念，這是「補時間」的機制



## 3. Tokenizer 與 Token 的角色
- LLM 並不真正「理解句子」而是「理解 token 之間的機率關係」
- **Tokenizer**：將文字轉換為模型可處理的 token（片段）
	- 例如：「I eat apples.」 → 4 tokens




## 4. Open Source vs Closed Source LLMs

|分類|優點|挑戰|
|---|---|---|
|**開源模型**|透明、可客製化、社群支援|建置難度高、需硬體與維護|
|**封閉模型**|穩定、易用、功能強大|成本高、隱私風險、無法自訂、供應商綁定|


## 5. Context & Memory Management

Context（上下文） ：指「與當前對話相關的資訊與狀態」，例如你上一句問了什麼、系統有什麼設定等。
Memory Management（記憶管理） ：指「儲存、提取、管理上下文」的機制，讓 AI 能夠連貫地對話，並在多輪對話中維持一致性。

**為什麼很重要？（Why It Matters）**

-   Coherence（連貫性）：保持回應一致、不跳脫主題
-   Personalization（個人化）：記得使用者偏好與歷史互動
-   Efficiency（效率）：重複使用已有的上下文，減少重新處理的資源浪費

OpenAI API 是怎麼處理 Context 的？

-   使用 `messages` 陣列儲存整個對話歷史
-   每筆訊息格式為：

```python
messages = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi there! How can I help you?"}
]
```

### 5.1. Context Window（上下文視窗）

一個 **可用 Token 的最大記憶容量**
當訊息太多時，會：

-   Truncate（截斷）舊訊息
-   或 Summarize（摘要化）過去的對話

```python
##### 使用llama，增加上下文記憶 #####
import ollama

# 對話上下文初始化
messages = [
    {"role": "user", "content": "你有靈魂嗎"},
]

# 儲存 AI 回覆的內容
assistant_response = ""

# 執行 stream 模式
res = ollama.chat(
    model="llama3.2:latest",
    messages=messages,
    stream=True,
)

# 一邊印出、一邊收集 AI 的回答
for chunk in res:
    content = chunk["message"]["content"]
    print(content, end="", flush=True)
    assistant_response += content

# 把 AI 的回覆加入 messages，維持上下文記憶
messages.append({"role": "assistant", "content": assistant_response})
```

## 6. LLM Logging

Logging 是在應用程式執行期間，紀錄事件、行為與資料的過程。
用途：

1. **Debugging 與錯誤追蹤**
    - LLM 是不可預測、複雜系統
    - Log 可以幫助追蹤使用者輸入、系統流程與錯誤訊息
2. **效能監控**：回應時間、token 使用量、錯誤率（error rate）
3. **法規遵循與稽核（Compliance & Auditing）**
4. **提升使用者體驗（UX）**：觀察回應品質、偏好、互動歷史
5. **安全性**：記錄可用來檢查是否有惡意使用或濫用模型

### 6.1. Logging in LLM Applications（架構流程）

```
User Input
   ↓
LLM Application
   ├── Log User Input
   ├── Log Model Response
   ├── Log Errors & Exceptions
   └── Log Performance Metrics
             ↓
          Log Storage（儲存在檔案、資料庫或雲端）
```

### 6.2. Logging Lifecycle（日常運作生命週期）

```
User Interaction
   ↓
Log Events（事件記錄）
   ↓
Store Logs（儲存 log）
   ↓
Analyze Logs（分析記錄）
   ↓
Identify Issues（找出問題）
   ↓
Fix & Improve（修正並改進）
   ↓
Better UX（使用者體驗提升）
```

### 6.3. Lab：與 ollama 交互，並 logging

```python
import logging
import json
from datetime import datetime
import uuid
import ollama

# ====== Step 1: 建立 Logging 設定 ======
def setup_logging():
    # 建立一個 logger 物件，名字叫 "structured_logger"
    logger = logging.getLogger("structured_logger")
    logger.setLevel(logging.INFO)  # 記錄等級設為 INFO（會記錄 info 以上）

    # 若 logger 已經有 handler，就清空（避免重複輸出）
    if logger.hasHandlers():
        logger.handlers.clear()

    # ➊ 檔案輸出：log 紀錄會寫到 chat_logs.json
    file_handler = logging.FileHandler("chat_logs.json")
    file_formatter = logging.Formatter('%(message)s')  # 只寫出純 JSON 內容
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # ➋ 終端機輸出：log 也會印到螢幕上（時間 + 訊息）
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger  # 回傳這個設定好的 logger


# ====== Step 2: 主聊天流程 ======
def main():
    logger = setup_logging()  # 初始化 logger
    session_id = str(uuid.uuid4())  # 每次啟動給一個唯一對話 session 編號

    print("💬 使用 Ollama 並紀錄完整對話 (輸入 'exit' 結束)")

    # 初始化 messages，並設定 AI 助理的角色
    messages = [
        {"role": "system", "content": "你是一位親切的 AI 助理，會用繁體中文回答。"}
    ]

    while True:
        # 等待使用者輸入
        user_input = input("你：").strip()

        # 若輸入 exit，離開程式
        if user_input.lower() == "exit":
            print("👋 結束紀錄，再見！")
            break

        # 空字串就跳過
        if not user_input:
            continue

        # 加入 user 輸入到上下文中
        messages.append({"role": "user", "content": user_input})

        # 記錄請求開始時間
        start = datetime.now()

        try:
            # 呼叫 Ollama，傳入目前所有上下文 messages
            response = ollama.chat(
                model="llama3.2",
                messages=messages
            )
        except Exception as e:
            print("⚠️ Ollama 回應失敗：", str(e))
            continue  # 跳過這輪輸入

        # 記錄請求結束時間
        end = datetime.now()

        # 取得 AI 回覆的文字
        ai_text = response["message"]["content"]
        print(f"AI：{ai_text}\n")

        # 把 AI 的回應也加入 messages（維持記憶）
        messages.append({"role": "assistant", "content": ai_text})

        # 建立一筆完整 log（使用者輸入 + 模型回覆 + 時間 + session ID）
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",  # UTC 時間標準
            "session_id": session_id,                          # 當前對話 ID
            "user_input": user_input,                          # 使用者的問題
            "model_response": ai_text,                         # 模型的回答
            "response_time": round((end - start).total_seconds(), 2),  # 回覆秒數
            "tokens_used": None  # ⚠ Ollama 沒有提供 tokens，所以預設為 None
        }

        # 輸出到 log 檔案與終端機
        logger.info(json.dumps(log_entry, ensure_ascii=False))


# ====== 主程式進入點 ======
if __name__ == "__main__":
    main()
```

## 7. Transformer Library（如 HuggingFace）

| 元件         | 說明                                   |
| ---------- | ------------------------------------ |
| Models     | 預訓練模型（如 GPT、BERT、T5）                 |
| Tokenizers | 支援多種分詞器（BPE、WordPiece、SentencePiece） |
| Pipelines  | 幫你快速套用模型到各種常見任務，不用寫太多程式              |

模型選擇的關鍵指標（Decision Factors）
- **預算限制（Budget）** 
- **資料隱私需求（Privacy）**
- **技術資源（Engineering）**
- **效能需求（Latency/Throughput）**
- **是否需要自訂模型行為（Customization）**

### 7.1. Lab：使用hugging face上的Transformer pipline進行文字生成
要先在hugging face上去的access token，會在huggingface-cli login要輸入
使用模型：[uer/gpt2-chinese-cluecorpussmall · Hugging Face](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)
```python
# 安裝
!pip install transformers torch
!pip install huggingface_hub

# 在終端機輸入
# huggingface-cli login 
```

```python
# 使用模型uer/gpt2-chinese-cluecorpussmall 

from transformers import pipeline

# 使用開放中文 GPT2 模型（適合中文生成）
generator = pipeline("text-generation", model="uer/gpt2-chinese-cluecorpussmall", device=0)

prompt = "從前從前有一位勇敢的女孩，她決定踏上旅程去尋找"
output = generator(prompt, max_length=100, do_sample=True, temperature=0.8)

print(output[0]["generated_text"].replace(" ", "").strip())
```

## 8. 使用 ChatGPT API 開發應用


API（Application Programming Interface ）： 讓你的程式可以**呼叫其他人的模型/服務**，不必自己訓練
舉例：
- 你的網頁使用者輸入問題
- 後端將問題發送至 OpenAI API
- API 回傳回答，前端顯示出來
好處
- 快速開發：不必訓練模型 
- 強大工具接入：像 GPT-4、LLaMA2、Claude 等
- 容易擴展：多使用者、多服務整合


| 步驟              | 說明                                       |
| --------------- | ---------------------------------------- |
| i. 設定環境         | 建立 Python 專案，安裝 `openai` 或 `requests` 套件 |
| ii. API 金鑰管理    | 申請並保管好 API key（例如 ChatGPT 的 OpenAI Key）  |
| iii. 第一次 API 呼叫 | 測試基本的 `POST` 請求送入 prompt                 |
| iv. prompt 測試   | 試不同問題、調整回應格式                             |
| v. 成本監控         | 控制 token 數量、管理帳單成本（尤其是 GPT-4）            |



### 8.1. Lab：使用 OpenAI API 呼叫 ChatGPT 模型
要先申請好open api，放在.env中

```python
from openai import OpenAI 
import os

from dotenv import load_dotenv

load_dotenv()  # 讀取 .env 中的變數
client = OpenAI()


# 建立聊天請求
response = client.chat.completions.create(
    model="gpt-4o",  # 使用的模型（你也可以改成 gpt-4、gpt-4o、gpt-4o-mini、gpt-3.5 等）
    messages=[
        {"role": "system", "content": "你是一位東方詩人。"},  # 指定系統角色為詩人
        {
            "role": "user",
            "content": """請寫一首關於月亮的短詩。
               使用俳句（Haiku）的風格來寫。
               請記得加上詩的標題。""",  # 使用者的請求內容
        },
    ],
)

print(response.choices[0].message.content)
```
![upgit_20250414_1744642588.png|658x424](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250414_1744642588.png)

## 9. Ollama 
**Ollama 是一個命令列工具（CLI）**，用來在本地端安裝與執行 LLM 模型。
目標：簡化本地部署 LLM 的流程。
類似於 Hugging Face + Docker 的結合體，但更簡化、更針對 LLM 使用。


### 9.1. 核心特點

- **Model Management**：集中管理多個模型
- **Unified Interface**：統一的 CLI 操作介面
- **Extensibility**：容易擴充功能
- **Performance Optimization**：針對硬體效能最佳化


### 9.2. Ollama 的底層流程簡介
相關：[【Day 26】- Ollama: 革命性工具讓本地 AI 開發觸手可及 - 從安裝到進階應用的完整指南 - iT 邦幫忙::一起幫忙解決難題，拯救 IT 人的一天](https://ithelp.ithome.com.tw/articles/10348913)

1. 使用者下達 **query**（問題）
2. 文件被分割為 **Chunks**
3. 使用 **Embedding LLM** 將所有 chunk 向量化
4. 找出最相似的片段（Top-K retrieval）
5. 傳入生成式 LLM，回傳 **回應（response）**

![upgit_20250414_1744646186.png|787x378](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250414_1744646186.png)

### 9.3. 解決什麼問題？

| 問題      | Ollama 解法         |     |
| ------- | ----------------- | --- |
| ✅ 隱私問題  | 模型在本地執行，不需將資料送上雲端 |     |
| ✅ 部署困難  | 一行指令就能執行 LLM      |     |
| ✅ 成本高   | 無須訂閱雲端 API        |     |
| ✅ 延遲高   | 本地模型，幾乎零延遲        |     |
| ✅ 客製化困難 | 可自行選擇模型並進行微調      |     |


### 9.4. 基本指令
```
# 查看所有指令說明
ollama help 

# 查看所有已下載模型
ollama list

# 安裝指令模型
ollama run llama3:3b

# 刪除某個模型（釋放空間）
ollama remove llama3:3b

##### 聊天模式指令 #####

# 使用某個 prompt 啟動（例如系統角色）
ollama run llama3 --system "你是一位熱情的數學家"

# 清除聊天上下文（重新開始）
/clear

/help

# 模型資訊
/show info
```

```
>>> /show info
  Model
    architecture        llama # 模型架構
    parameters          3.2B # 參數量：代表模型的大小與學習能力。
    context length      131072 # 上下文處理長度
    embedding length    3072 #  詞嵌入維度(數值越大代表模型能夠更細緻地理解語意)
    quantization        Q4_K_M #  量化方式
```

### 9.5. 自定義模型設定檔 Modelfile
Modelfile 是什麼：有點像 Ollama 的「設定檔」，可讓你基於一個現有模型（如 llama3）定義
- 預設角色、語氣
- 溫度（創造力）
- 初始 prompt
- 指定 tokenizer 或其他設定

STEP01：建立檔案【Modefile】
```
# 指定你要基於哪一個模型來建立新的模型
FROM llama3.2:latest

##### 參數(PARAMETER) #####
PARAMETER temperature 0.3 # 創意程度
PARAMETER top_p 0.9
# 最大生成長度
PARAMETER num_predict 1024
# 防止重複出現相同詞。建議值：1.1 ~ 1.3
PARAMETER repeat_penalty 1.2

##### SYSTEM：設定 AI 助理的角色/語氣 ##### 
SYSTEM """
你是一位專業的老師，擅長用淺顯例子解釋複雜概念。
"""

##### LICENSE / MESSAGE / MODIFIER（可選補充）##### 
# 顯示在 ollama list 中的資訊提示
LICENSE MIT
MESSAGE "本模型用途為教育研究，請勿濫用"
```

STEP02：使用【Modefile】建立模型：
```
ollama create my-teacher -f Modelfile
```
STEP03：使用模型
```
ollama run my-teacher
```

### 9.6. 讓 Ollama 變成 REST API，並使用python進行交互
啟動 Ollama 後，它會開一個本地端 API Server：
```
http://localhost:11434 => 只要啟動了 Ollama，就可以用 POST 方式呼叫這個 API。
# 相關應用，使用 Python 呼叫 Ollama API
```

```python
# 使用 Python 呼叫 Ollama API
import requests
import json

url = "http://localhost:11434/api/generate"

payload = {
    "model": "llama3.2:latest",
    "prompt": "請用一段話介紹 Arduino 是什麼。",
    "stream": False
}

response = requests.post(url, json=payload, stream=True)
print(response.json()["response"])
```

#### 9.6.1. 如果你想從「別的電腦」存取 Ollama（變成 Web 服務）？
- 找到你跑 Ollama 主機的 IP，例如：192.168.1.100
- 開放防火牆與埠號（11434）
- 後其他機器就可以透過`http://192.168.1.100:11434/api/generate`連線

#### 9.6.2. ✅ Ollama 所有常用 REST API Endpoint 整理

|Endpoint|方法|用途說明|
|---|---|---|
|`/api/generate`|POST|✅**產生文字回應**（類似 ChatGPT 回應）|
|`/api/chat`|POST|✅**具對話上下文的聊天模式**（多輪對話）|
|`/api/models`|GET|查看已安裝的模型列表|
|`/api/pull`|POST|✅ 下載（pull）模型|
|`/api/create`|POST|建立自定模型（配合 `Modelfile` 使用）|
|`/api/delete`|DELETE|刪除已安裝的模型|
|`/api/embeddings`|POST|產生文字的向量（embedding，用於搜尋、分類等）|
|`/api/stop`|POST|手動停止生成中請求（例如使用者點取消）|

### 9.7. 搭配搭配ollama使用的圖形介面APP：msty
[Msty - Using AI Models made Simple and Easy](https://msty.app/)

### 9.8. python交互llama庫

```python
#### 基本聊天 #####
import ollama

res = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {"role": "user", "content": "你有靈魂嗎"},
    ],
)

print(res["message"]["content"])
```

```python
#### Chat example streaming #####
import ollama

res = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {"role": "user", "content": "你有靈魂嗎"},
    ],
    stream=True,
)

for chunk in res:
    print(chunk["message"]["content"], end="", flush=True)
```

```python
#### Create a new model with modelfile #####
import ollama

modelfile = """
FROM llama3.2:latest
PARAMETER temperature 0.4
SYSTEM 你是一位專業的老師，擅長用淺顯例子解釋複雜概念。
"""

ollama.create(model="knowitall", modelfile=modelfile)

res = ollama.generate(model="knowitall", prompt="why is the ocean so salty?")
print(res["response"])

ollama.delete("knowitall") # delete model
```