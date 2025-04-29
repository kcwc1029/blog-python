LangChain 是一個開源（open source）的框架，用來快速建構結合 多種 LLMs 的應用程式。
除了內建 LLM 外，也可以連結外部資料來源（external sources of data）！


## 1. LLM Frameworks（大型語言模型開發框架）

為什麼需要 LLM 框架？
- 簡化整合：專注於開發應用，而不用自己管理底層細節
- 模組化設計：拆分成可重用的元件（易於擴展和維護）
- 支援上下文管理：模型需要記憶上下文（提供記憶機制）
- 方便資料整合：接 API、資料庫很簡單
- 加速開發流程：開發智能代理（Agents）等高階功能更快
    

常見 LLM Frameworks：
- LangChain（通用型，支持記憶、代理、模組化流程）
- LlamaIndex（重視 RAG 與結構化/非結構化資料檢索）
- Haystack（重視搜尋與問答系統建置）
    

## 2. 🔑 LangChain 核心組件（Key Components）

- Models：包裝不同模型接口（GPT, Embedding model, Chat model）
- Prompts：統一管理提示詞、支援範例學習（few-shot learning）
- Chains
    - 支援串連多個步驟：如先問資料庫再問 LLM        
    - 支援自定義工作流程（Custom Chains）
- Memory
    - 短期記憶：只記住單次對話
    - 長期記憶：跨多輪對話保持上下文
- Tools：提供外部工具整合，例如 API 資料來源、資料庫存取        
- Agents：自主系統，讓模型能根據需求選擇行動，不只回覆問題





## 3. 📈 LangChain 實際應用場景

- 文件問答系統（Document QA）
- 智能對話機器人
- 企業內部知識庫檢索
- 自動化資料處理代理人
- 資料蒐集與外部 API 整合

### 3.1. Lab：使用LangChain與deepseek進行交互基本問答

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# DeepSeek API KEY
import os
api_key = os.getenv("DEEPSEEK_API_KEY")  # 你的 .env 要存這個 API KEY

# 初始化 DeepSeek
model = ChatOpenAI(
    openai_api_key=api_key,
    base_url="https://api.deepseek.com",  
    model="deepseek-chat",  
)

# 建立訊息
messages = [
    SystemMessage(content="將下列的英文翻譯成中文"),
    HumanMessage(content="hi!"),
]

# 呼叫
response = model.invoke(messages)
print(response.content)
```

### 3.2. Lab：使用LangChain建立 **Prompt Template**（有變數的提示詞）
```python
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

system_template = "將下列英文翻譯成 {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "中文", "text": "hi"})
print(prompt)
response = model.invoke(prompt)
print(response.content)
```


## 4. Indexes、Retrievers 與 Data Preparation

### 4.1. Data Preparation（資料準備）

**目標**：  把**原始資料**轉成可以給 AI 檢索的結構化資料。

**流程**：
- **Raw Data**（原始資料）
- ➡️ **Document Loader**（讀取文件）
- ➡️ **Text Splitters**（文字切割器，把長文件切小段）
- ➡️ **Text Chunks**（一小段一小段的文本）
- ➡️ **Embeddings**（轉成向量）
- ➡️ 存入 **Vector Store**（向量資料庫）
    

### 4.2. Indexes（索引）

**目標**：  快速組織和整理文件，方便快速查詢。
把文件整理成「索引」，就像圖書館建目錄一樣，方便找資料。
**常見索引類型**：
- **Vector Store Index**  
    → 把文件轉成向量，可以用「相似度」來快速找資料。  
    → 範例：FAISS、Pinecone、Chroma。
- **List Index**  → 把所有文件簡單地列成一個清單。
- **SQL Index**  → 用 SQL 方式儲存結構化資料。
- **Graph Index**  → 用「圖形結構」儲存資料，比如知識圖譜。

### 4.3. Retrievers（檢索器）
Retrievers 就像圖書館櫃檯員，幫你根據「問題」快速找到最相關的資料。
**目標**：  根據用戶輸入的問題，找到最相關的文件段落。
**Retrievers 的功能**：
- 收到 **Query**（問題）
- ➡️ 進行檢索（根據索引資料庫）
- ➡️ 找到 **Relevant Documents**（相關文件）

**Retrievers 常見方式**：
- **Vector Search**（向量相似度搜尋）
- **Keyword Search**（關鍵字搜尋）
- **Hybrid Search**（混合搜尋）


### 4.4. Full Workflow（完整流程）
1. 原始資料
2. ➡️ Document Loader（讀入）
3. ➡️ Text Splitters（切割）
4. ➡️ Text Chunks（小段文字）
5. ➡️ Embeddings（向量轉換）
6. ➡️ Vector Store（向量資料庫）
7. ➡️ Query（提問）
8. ➡️ Retriever（搜尋相關段落）
9. ➡️ LLM（大語言模型回答）
10. ➡️ Response（產生回覆）
    

### 4.5. Lab：LangChain Text Splitter
怎麼把一份大文件，切成很多小段落 (chunks)
每段都控制在一定大小內，方便之後給模型使用！
這是做 RAG (Retrieval-Augmented Generation) 的前置步驟！
```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load your document 
text_loader = TextLoader("./dream.txt")  
documents = text_loader.load()  

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

# Split documents
splits = text_splitter.split_documents(documents)
for i, split in enumerate(splits):
    print(f"Split {i+1}:\n{split}\n")
```

```text
# 輸出效果
Split 1:
page_content='And so even though we face the difficulties of today and tomorrow, I still have a dream. It is a' metadata={'source': './dream.txt'}

Split 2:
page_content='a dream. It is a dream deeply rooted in the American dream.' metadata={'source': './dream.txt'}

Split 3:
page_content='I have a dream that one day this nation will rise up and live out the true meaning of its creed:' metadata={'source': './dream.txt'}
...
```
### 4.6. Lab：LangChain Text Loaders
安裝套件
```
pip install faiss-cpu
```

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import re
import os


api_key = os.getenv("DEEPSEEK_API_KEY")  # 你的 .env 要存這個 API KEY

# 清理文字資料（Data Cleaning）
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # 只留英文字母和空白
    text = re.sub(r"\s+", " ", text).strip()  # 把多餘的空白合成一個空白
    text = text.lower()                      # 全部轉小寫
    return text

# 載入文字檔案
documents = TextLoader("./dream.txt").load()
cleaned_documents = [clean_text(doc.page_content) for doc in documents] # 讀取 dream.txt 這個檔案內容。

# 切割文字成小段（分段）
# chunk_size=500：每段最多500個字     
# chunk_overlap=100：前後段落重疊100個字。
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) 
texts = text_splitter.split_documents(documents)
texts = [clean_text(text.page_content) for text in texts]

# 把小段轉成向量 Embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 建立檢索器 Retriever
retriever = Chroma.from_texts(texts, embeddings).as_retriever()

# 查詢 retriever
query = "請以精要點概述演講內容"
docs = retriever.invoke(query) # 最有可能回答你問題的文件段落集合
# print(docs)

# Chat with the model and our docs
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(
    "Please use the following docs {docs},and answer the following question {query}",
)


# 初始化 DeepSeek
model = ChatOpenAI(
    openai_api_key=api_key,
    base_url="https://api.deepseek.com",  # DeepSeek官方API URL
    model="deepseek-chat",  # 也可以是 deepseek-coder
)
chain = prompt | model | StrOutputParser()

response = chain.invoke({"docs": docs, "query": query})
print(response)
```


### 4.7. Lab：LangChain PDFLoader
```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
)

# 載入單個PDF
pdf_loader = PyPDFLoader("./data/DeepSeek-Coder When the Large Language Model Meets.pdf")

docs = pdf_loader.load()
print("PDF Documents:", docs)
```


### 4.8. Lab：LangChain DirectoryLoader
要先做一個資料夾【data】，然後他是將【data】內所有文件載入
```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, DirectoryLoader

dir_loader = DirectoryLoader("./data/", glob="**/*.txt", loader_cls=TextLoader)
dir_documents = dir_loader.load()

print("Directory Text Documents:", dir_documents)
```

### 4.9. Lab：chains
如何把多個處理步驟「串接」起來，像一條鏈一樣，把「提示（prompt）→模型推論→輸出解析」這些步驟整合成一個流程。

這邊是使用langchain連接deepseek，所以要先安裝套件
```python
pip install langchain-deepseek
```

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv("DEEPSEEK_API_KEY")  # 你的 .env 要存這個 API KEY


# Define a prompt template
prompt = ChatPromptTemplate.from_template("我今天心情{mood}")

# Create a chat model：初始化 DeepSeek
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1024,
    timeout=30,
    max_retries=2,
)

# Chain the prompt, model, and output parser
chain = prompt | model | StrOutputParser()

# Run the chain
response = chain.invoke({"mood": "真的非常糟"})
print(response)
```

### 4.10. Lab：LangChain 架構下，典型的 RAG（Retrieval-Augmented Generation）流程
流程
- 要準備一份txt，作為檢索文件，這邊測試的文件檔名為dream.txt
- 讀取 + 清洗 (clean_text)
- 分段 chunk（每段500字）
- 向量化 embedding（text-embedding-3-small）
- 建立向量資料庫 Chroma + 檢索器 retriever
- 使用者提問 query
- retriever 擷取相關段落
- prompt 填入 {docs} + {query}
- 組合提示 + 啟動 DeepSeek 模型回答

```python
from langchain_community.document_loaders import TextLoader        # 讀取本地文字檔
from langchain_community.vectorstores import Chroma                # 使用 Chroma 作為向量資料庫
from langchain_openai import OpenAIEmbeddings                     # 使用 OpenAI 的向量嵌入模型
from langchain_deepseek import ChatDeepSeek                       # 使用 DeepSeek 聊天模型
from langchain_text_splitters import RecursiveCharacterTextSplitter # 將長文本切割成小段落
from langchain_core.output_parsers import StrOutputParser         # 將模型回應轉換為純字串
from langchain_core.prompts import ChatPromptTemplate             # 建立聊天提示模板
from dotenv import load_dotenv                                     # 載入 .env 設定檔（例如 API 金鑰）
import os
import re                                                      


# 載入環境變數
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")  # 從 .env 讀取 DeepSeek API 金鑰


# 定義文字清理函數
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)     # 移除非英文字母與空白
    text = re.sub(r"\s+", " ", text).strip()    # 移除多餘空白
    return text.lower()                            # 全部轉為小寫


# 載入並處理文件
documents = TextLoader("./dream.txt").load()                      # 載入文字文件
documents = [clean_text(doc.page_content) for doc in documents]   # 清洗每段內容

# 分割文字為多段 chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.create_documents(documents)                # 切割後轉換為 Document 格式
chunks = [clean_text(doc.page_content) for doc in chunks]         # 再次清洗每個 chunk


# 向量化文本並建立檢索器
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="my_documents",
    # persist_directory="./chroma_db"  # 如果需要持久化儲存，可取消註解
).as_retriever(search_kwargs={"k": 2}) # 取回與查詢最相近的前 2 段文字


# 使用者問題 + 檢索相關內容 
query = "這篇文章的核心思想是?"
docs = retriever.invoke(query)   # 根據 query 從 Chroma 檢索相關內容

# 模型回應
prompt = ChatPromptTemplate.from_template(
    "請根據文件 {docs}, 並使用中文回答以下問題 {query}"
)

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1024,
    timeout=30,
    max_retries=2,
)

chain = prompt | model | StrOutputParser()     # 組合 prompt、模型、解析器為一條鏈
response = chain.invoke({"docs": docs, "query": query})     # 執行鏈並取得模型回應


print(f"Model Response:::\n{response}")
```