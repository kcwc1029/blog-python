

Prompt：Prompt 是你給 LLM 的輸入文字，告訴模型你想得到什麼樣的回應。

為何 Prompt Engineering 很重要？

-   **提升結果品質**：更精準、符合需求的回應
-   **減少誤解**：指令明確可避免偏離主題
-   **提高控制力**：引導模型行為更符合應用目的

## 1. Prompt 類型

| 類型                        | 用途                         | 範例                                             | 延伸說明                                                                                 |
| --------------------------- | ---------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **Direct Prompt**           | 查詢事實、快速回答           | 「台灣的首都是？」                               | - 結構最簡單，常見於問答類應用- 不含情境、角色、細節，適合 Chatbot FAQ                   |
| **Open-ended Prompt**       | 生成開放性、多樣化回應       | 「請寫一篇關於龍與騎士的故事」                   | - 回應可能高度多變，適合創意寫作、構思靈感- 可搭配 "以三段式結構撰寫" 增加結構性         |
| **Instructional Prompt**    | 給予明確任務與要求格式       | 「請列出 Python 的優缺點，以清單方式整理」       | - 適合摘要、筆記、報告生成等工作- 可搭配格式指令（如 bullet point、表格）                |
| **Role-based Prompt**       | 指定語氣、知識背景與視角     | 「你是一位歷史學家，請解釋第一次世界大戰的原因」 | - 能改變模型回答方式，強化說服力與專業感- 適用於模擬專家、客服、導師、編輯等情境         |
| **Chain-of-thought Prompt** | 引導模型逐步推理，增強邏輯性 | 「請一步一步解釋 3x + 5 = 20 的解法」            | - 適合數學、邏輯、推理題目- 可以幫助模型「慢思考」，提升正確率（特別是數學與多步驟任務） |

## 2. Prompt 撰寫原則

|                                            |                  |                                               |                                                                                                                |
| ------------------------------------------ | ---------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Clarity**（清晰）                        | 明確說明任務     | 寫一篇文章                                    | 請寫一篇 300 字的文章，主題是永續旅遊，包含至少三個策略                                                        |
| **Specificity**（具體）                    | 避免模糊語句     | 總結這篇文章                                  | 請用三個重點句，摘要這篇關於 ChatGPT 應用於教育的文章內容                                                      |
| **Contextualization**（上下文）            | 提供足夠背景資訊 | 解釋這段程式碼                                | 你是一位 Python 老師，請用高中生能理解的方式，逐行解釋以下的迴圈語法                                           |
| **Leverage Examples**（給示範）            |                  |                                               | 請模仿以下格式回覆：\n- 優點：...\n- 缺點：...\n 範例：ChatGPT 的優缺點為...                                   |
| **Iterative Refinement**（反覆調整與測試） |                  | 初稿 prompt：「解釋記憶體管理」→ 回應過於簡略 | 修正為：「請以大學生能理解的程度，並包含三個例子，說明作業系統中的記憶體管理機制」→ 回應變得結構清楚、條理分明 |

## 3. Prompt 工程應用策略（進階）

### 3.1. 策略 01：Few-shot Prompting

給模型數個示範例子
增強模型理解你要的輸出風格

```python
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()  # 讀取 .env 中的變數
client = OpenAI()


# 建立聊天請求
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一位中文詩人，擅長寫關於自然的俳句。"},
        {"role": "user", "content": "請幫我寫一首關於春天的俳句。"},
        {"role": "assistant", "content": "《春之聲》\n\n櫻花初綻，\n微風拂過枝頭，\n燕子歸來。"},
        {"role": "user", "content": "請幫我寫一首關於秋天的俳句。"},
    ]
)
print(response.choices[0].message.content)
```

### 3.2. 策略 02：Zero-shot Prompting

不給範例，直接下指令
GPT-4/Claude 這類強模型也能處理好

```python
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()  # 讀取 .env 中的變數
client = OpenAI()


# 建立聊天請求
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一位中文詩人。"},
        {"role": "user", "content": "請用俳句風格，寫一首描述夏夜星空的詩，並加上詩的標題。"}
    ]
)
print(response.choices[0].message.content)

```

### 3.3. 策略 03：Instruction Tuning

使用指令式語句設計 prompt
LLM 如 FLAN-T5、ChatGPT 針對此優化

```python
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()  # 讀取 .env 中的變數
client = OpenAI()


# 建立聊天請求
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是知識豐富的教育助理。"},
        {"role": "user", "content": "請列出三個 Python 的優點，並用條列式整理。"}
    ]
)
print(response.choices[0].message.content)
```

### 3.4. 策略 04：Output Formatting

指定輸出格式，如「以 JSON 回覆」
對資料整理、前端應用非常實用

```python
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()  # 讀取 .env 中的變數
client = OpenAI()


# 建立聊天請求
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是格式非常嚴謹的 AI 小幫手。"},
        {"role": "user", "content": "請以 JSON 格式輸出一則包含標題、內容與作者的短文。"}
    ]
)
print(response.choices[0].message.content)

```

### 3.5. 策略 05：多語言輸入控制

用 `請用繁體中文回覆` 來語言限定
特別適合教學語境或特定場域應用

```python
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()  # 讀取 .env 中的變數
client = OpenAI()


# 建立聊天請求
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是中文教師。"},
        {"role": "user", "content": "請用繁體中文解釋什麼是人工智慧，並舉一個生活中的例子。"}
    ]
)
print(response.choices[0].message.content)

```

