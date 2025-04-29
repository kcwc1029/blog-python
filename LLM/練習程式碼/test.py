# --- 載入需要用的工具 ---
from typing import Dict, List  # 型別提示：讓別人或自己更清楚這裡是用什麼資料型別
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 用來把長文章切成小段
from langchain_openai import OpenAIEmbeddings  # 把文字轉成向量（給電腦理解的數字）
from langchain_community.vectorstores import Chroma  # 向量資料庫（存文字向量的地方）
from langchain_openai import ChatOpenAI  # OpenAI 的聊天模型（像是 GPT）
from langchain_community.document_loaders import SeleniumURLLoader  # 用 Selenium 去抓網頁文章
from langchain_core.output_parsers import StrOutputParser  # 把 GPT 回答整理成純文字
from langchain_core.runnables import RunnablePassthrough  # 幫助資料流程傳遞
from langchain_core.prompts import ChatPromptTemplate  # 自訂提示詞格式
from dotenv import load_dotenv  # 讀取 .env 檔案，設定 API 金鑰等

# --- 載入環境變數 ---
load_dotenv()

# --- 設定模型名稱 ---
model_name = "gpt-4o-mini"

# --- 要抓取的網頁網址 ---
documents = [
    "https://beebom.com/what-is-nft-explained/",
    "https://beebom.com/how-delete-servers-discord/",
    "https://beebom.com/how-list-groups-linux/",
    "https://beebom.com/how-open-port-linux/",
    "https://beebom.com/linux-vs-windows/",
]

# --- 抓取網頁內容的函式 ---
def scrape_docs(urls: List[str]) -> List[Dict]:
    """用 SeleniumURLLoader 抓取網頁內容"""
    try:
        loader = SeleniumURLLoader(urls=urls)
        raw_docs = loader.load()
        print(f"\n成功載入 {len(raw_docs)} 篇文章")
        for doc in raw_docs:
            print(f"\n來源: {doc.metadata.get('source', '無來源')}")
            print(f"內容長度: {len(doc.page_content)} 字元")
        return raw_docs
    except Exception as e:
        print(f"抓取文章時出錯: {str(e)}")
        return []

# --- 把文章切成小段的函式 ---
def split_documents(pages_content: List[Dict]) -> tuple:
    """將每篇文章切成小塊文字"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    all_texts, all_metadatas = [], []
    for document in pages_content:
        text = document.page_content  # 抓文章文字
        source = document.metadata.get("source", "")  # 抓來源網址

        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadatas.append({"source": source})

    print(f"切出了 {len(all_texts)} 段小文字")
    return all_texts, all_metadatas

# --- 建立向量資料庫的函式 ---
def create_vector_store(texts: List[str], metadatas: List[Dict]):
    """建立向量資料庫（Chroma）"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_texts(texts=texts, metadatas=metadatas, embedding=embeddings)
    return db

# --- 建立問答流程（QA Chain）的函式 ---
def setup_qa_chain(db):
    """設定問答機器人流程"""
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    retriever = db.as_retriever()

    # 客製化有禮貌的問答提示模板
    prompt = ChatPromptTemplate.from_template(
        """
        請根據以下提供的內容，給出一個有禮貌且具幫助性的回答。
        保持專業、清楚、積極的語氣。

### 內容:
{context}

### 問題: 
{question}

### 請有禮貌地回答:
"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# --- 處理提問的函式 ---
def process_query(chain_and_retriever, query: str):
    """拿問題去問，並且回應答案"""
    try:
        chain, retriever = chain_and_retriever

        response = chain.invoke(query)
        docs = retriever.invoke(query)
        sources_str = ", ".join([doc.metadata.get("source", "") for doc in docs])

        return {"answer": response, "sources": sources_str}
    except Exception as e:
        print(f"處理問題時出錯: {str(e)}")
        return {
            "answer": "抱歉，處理你的問題時出錯了。",
            "sources": "",
        }

# --- 主程式開始 ---
def main():
    # 1. 抓取文件
    print("正在抓取網頁文章...")
    pages_content = scrape_docs(documents)

    # 2. 切割文件
    print("正在切割文章...")
    all_texts, all_metadatas = split_documents(pages_content)

    # 3. 建立向量資料庫
    print("正在建立向量資料庫...")
    db = create_vector_store(all_texts, all_metadatas)

    # 4. 設定問答流程
    print("正在設定問答流程...")
    qa_chain = setup_qa_chain(db)

    # 5. 互動式問答
    print("\n可以開始問問題了！（輸入 'quit' 離開）")
    while True:
        query = input("\n請輸入你的問題: ").strip()

        if not query:
            continue
        if query.lower() == "quit":
            break

        result = process_query(qa_chain, query)

        print("\n回答：")
        print(result["answer"])

        if result["sources"]:
            print("\n參考來源：")
            for source in result["sources"].split(","):
                print("- " + source.strip())

# --- 程式啟動 ---
if __name__ == "__main__":
    main()


