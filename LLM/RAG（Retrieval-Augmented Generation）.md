




## 1. RAG基本知識
 RAG 是一種結合「檢索式系統」與「生成式模型」的框架，用於生成更準確且有上下文依據的回應。
「讓 LLM 模型可以『查資料』再『回答問題』的一種方法，彌補它只知道訓練資料的限制。」

> RAG = Retrieval + Augmentation + Generation

組成元件：
- Retriever：負責檢索與問題相關的資料
- Generator：根據檢索到的資料與問題生成回應


應用場景（Practical Applications）
- 🤖 Q&A 系統（e.g., Chatbot 可引用正確資訊回答）
- 🔍 搜尋強化（提升搜尋準確度與文字品質）
- 🧾 文件摘要（讀取、彙整長篇內容）
- 🎓 教育學習系統（以教科書資料回答學生問題）

優勢（Benefits）
- **Enhanced factuality**：引用真實資料
- **Improved accuracy & specificity**
- **Reduced hallucination**
- **No retraining needed**：不需重新訓練模型

面臨的挑戰（Challenges）
- ⚠️ 檢索品質：如果抓到的資料不準，會影響回應品質
- ⚠️ 運算成本高：多一步檢索與處理
- ⚠️ 整合難度：檢索與生成需調整 prompt、流程
- ⚠️ 知識庫偏誤：資料來源不公正會造成錯誤回答


RAG 與純生成模型的比較

| 模型 | 特點 | 問題 |
|------|------|------|
| ✅ RAG | 有檢索能力，回答具上下文依據 | 效率與系統設計需調整 |
| ❌ 純 LLM | 只靠訓練資料生成答案 | 可能幻覺（Hallucination）、過時、不準確 |



### 1.1. RAG 如何運作？
1. 使用者提出問題（Query）
2. Retriever 向資料庫中搜尋相關段落（chunks/documents）
3. 把檢索資料與原始問題合併成 prompt（Augmentation）
4. Generator 使用 LLM 產生回應



### 1.2. RAG 技術流程（Deep Dive）

🔧 文件 → 分段 → 向量化 → 儲存至向量資料庫 → 檢索相似向量 → 生成回答

1. 文本預處理與 Chunking
2. 使用 Embedding model 轉為向量
3. 存入 Vector DB（如 FAISS、Weaviate）
4. 使用者提問時進行向量查詢
5. 檢索回來的內容 + prompt 餵進 LLM


![upgit_20250416_1744806560.png|854x483](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250416_1744806560.png)

太棒了，你一次學完了大量與 **Vector Database（向量資料庫）** 與其在 **AI、RAG（檢索增強生成）** 中的應用相關概念，我幫你系統化地整理如下筆記 👇，完整又有邏輯順序，適合複習、教學或筆記保存。

---

## 2. 向量資料庫（Vector Database）與嵌入向量（Embeddings）

### 2.1. 向量（Vector）
- 是一組有方向（Direction）與大小（Magnitude）的數值  
- 在 AI 裡，向量是資料的「數學表示法」  
- 例子：文字、圖片、聲音可以被轉換成向量
### 2.2. 向量資料庫（Vector Database）
- 一種可用來儲存與比對 **高維向量** 的資料庫
- 可快速執行「相似度搜尋」
- 資料通常來自未結構化來源（如 PDF、圖片、音檔、文章）

### 2.3. 為什麼需要向量資料庫？
- 超過 80% 資料是「非結構化資料」
  - 如：圖片、音訊、影片、PDF、對話記錄等
- 傳統關聯式資料庫（如 MySQL）不擅長處理這些

![upgit_20250416_1744808228.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250416_1744808228.png)

![upgit_20250416_1744808242.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250416_1744808242.png)


### 2.4. 向量資料庫的強項
- 能將非結構化資料轉換為向量
- 支援高維空間的相似度查詢（比對語意）
- 適合 AI、RAG、搜尋與推薦系統應用



### 2.5. 向量資料庫的運作流程（Deep Dive）

1. 📂 收集未結構化資料（圖片、文字、PDF）
2. ✂️ **分割文本**（Text Splitting / Chunking）
3. 🧠 **嵌入（Embedding）模型** 將每個 chunk 轉為向量
4. 💾 存進向量資料庫（如 FAISS、Pinecone、Weaviate）
5. 🧑‍💻 使用者發問時：
   - 問題也會經過 embedding ➝ 得到 query 向量
   - 與資料庫中向量做比對（similarity search）
   - 找出最相近的幾筆資料
   - 提供給 LLM 生成答案

![upgit_20250416_1744808331.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250416_1744808331.png)

![upgit_20250416_1744808340.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250416_1744808340.png)

![upgit_20250416_1744808349.png](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250416_1744808349.png)



### 2.6. vector與 Embedding 差異？
>  **Embedding 是一種向量**，但更專注於語意的保留與搜尋應用。

| 向量（Vector） | 嵌入（Embedding） |
|----------------|-------------------|
| 一般意義的多維數值 | 特別為 AI 模型訓練而生成的語意向量 |
| 可用於各種數學計算 | 強調保留語意與關係（semantic similarity） |
| 廣泛泛用 | 專用於 AI / NLP 語意比對任務 |

### 2.7. 與傳統資料庫差異

| 傳統資料庫（RDBMS） | 向量資料庫 |
|---------------------|-------------|
| 結構化資料（表格） | 非結構化資料（文字、圖像） |
| 查詢用 SQL、index、key-value | 查詢用「語意相似度」 |
| 擅長篩選與關聯比對 | 擅長語意理解與模糊比對 |
| 不支援語意搜尋 | 可做 embedding-based 搜尋 |



### 2.8. 關鍵詞統整

| 名詞 | 定義 |
|------|------|
| **Embedding** | 將資料轉為向量的模型 |
| **Vector Store** | 儲存並提供相似比對的資料庫 |
| **Similarity Search** | 比較向量距離，找最相似資料 |
| **Query Vector** | 使用者的提問被轉成的向量 |
| **Chunking** | 將長文件切成短段的過程 |
| **ANN** | Approximate Nearest Neighbor（快速近似比對技術）|

## 3. Chroma Database workflow

![upgit_20250416_1744808629.png|714x402](https://raw.githubusercontent.com/kcwc1029/obsidian-upgit-image/main/2025/04/upgit_20250416_1744808629.png)

安裝套件
```
pip install chromadb
```

### 3.1. Lab：用 ChromaDB 建立一個本地向量資料庫並加入資料的最基本範例
```python
#####  Add some text documents to the collection #####
import chromadb

# 初始化 ChromaDB（會在本地建立 .chroma 資料夾）
client = chromadb.Client()

# 創建 collection（你可以視為一個向量表）
collection = client.create_collection(name="my_collection")

# 新增一筆資料（內部會用內建 Embedding function 處理）
collection.add(
    documents=["今天吃拉麵", "我喜歡貓咪", "Python 是好用的語言"],
    ids=["id1", "id2", "id3"]
)

##### Query the collection #####
results = collection.query(
    query_texts=["我想學貓咪相關的東西"],
    n_results=2  # 要回傳幾筆最相似的結果
)

print(results)
```

### 3.2. Lab：Chroma Default Embedding Function
把一段文字（像 "Paulo"）轉換成 向量表示（embedding）
```python
from chromadb.utils import embedding_functions

# 建立一個 DefaultEmbeddingFunction() 實例
default_ef = embedding_functions.DefaultEmbeddingFunction()

name = "Paulo"
emb = default_ef(name)
print(emb)
```

### 3.3. Lab：使用ersistentClient 建立「持久化向量資料庫」
```python
from chromadb.utils import embedding_functions

##### 初始化向量資料庫與嵌入器 ##### 
default_ef = embedding_functions.DefaultEmbeddingFunction()
croma_client = chromadb.PersistentClient(path="./db/chroma_persist")

##### 建立或讀取 collection（向量集合）#####
collection = croma_client.get_or_create_collection(
    "my_story", embedding_function=default_ef
)


##### 新增文件資料（upsert）#####
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
    {
        "id": "doc4",
        "text": "Microsoft is a technology company that develops software. It was founded by Bill Gates and Paul Allen in 1975.",
    },
]

for doc in documents:
    collection.upsert(ids=doc["id"], documents=[doc["text"]])

##### 語意查詢（semantic search）#####
query_text = "find document related to technology company"

results = collection.query(
    query_texts=[query_text],
    n_results=2,
)



##### 解析查詢結果 #####
print(results)
print()
for idx, document in enumerate(results["documents"][0]):
    print(document)
```

### 3.4. Lab：使用 OpenAI 的 Embedding API，將一段文字轉換成「向量（vector / embedding）」。

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.embeddings.create(
    input="好啦你最棒，你最清高，你是神仙", 
    model="text-embedding-3-small" # 使用的嵌入模型
)

print(response)
```

### 3.5. Lab：用 OpenAI 嵌入模型 + ChromaDB 建立語意搜尋系統（Semantic Search）
```python
import chromadb
import os
from dotenv import load_dotenv
from chromadb.utils import embedding_functions


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# 設定 embedding 函式與 ChromaDB
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key, model_name="text-embedding-3-small" # 設定好 用哪個模型 把文字轉向量（OpenAI 模型）
)

croma_client = chromadb.PersistentClient(path="./db/chroma_persist") # 建立一個可「持久儲存」的 ChromaDB 本地資料庫

# 建立 collection，並使用剛剛的 OpenAI embedding 方法
collection = croma_client.get_or_create_collection(
    "my_story", embedding_function=openai_ef # 建立my_story語意資料表，專門存你要查詢的語意向量
)

# 新增資料（會自動做 embedding）
documents = [
    {"id": "doc1", "text": "哈囉，世界！"},
    {"id": "doc2", "text": "你今天過得如何？"},
    {"id": "doc3", "text": "再見，待會見！"},
    {"id": "doc4", "text": "微軟是一家科技公司。"},
    {"id": "doc5", "text": "人工智慧模擬人類智慧。"},
    {"id": "doc6", "text": "機器學習能從資料中學習。"},
    {"id": "doc7", "text": "深度學習使用多層神經網路。"},
    {"id": "doc8", "text": "自然語言處理讓電腦理解人類語言。"},
    {"id": "doc9", "text": "AI 分為狹義與廣義兩種。"},
    {"id": "doc10", "text": "電腦視覺處理影像資料。"},
    {"id": "doc11", "text": "強化學習透過回饋學習行為。"},
    {"id": "doc12", "text": "圖靈測試判斷 AI 是否像人。"},
]


for doc in documents: # 把每一筆文字 + ID 加進資料庫
    collection.upsert(ids=doc["id"], documents=[doc["text"]])



##### 發送語意查詢 #####
# 將查詢文字轉成語意向量
# 跟資料庫比對距離（越小越近）
# 回傳最相近的三筆資料
query_text = "查找與圖靈測試相關的文件"
results = collection.query(query_texts=[query_text], n_results=3)
for idx, document in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    # distance = results["distances"][0][idx]
    print(document)
```

### 3.6. Lab：Loading all Documents
檔案：Project：Chroma 文件嵌入與查詢初始化專案
檔案：Project：RAG CSV
檔案：Project：RAG PDF



