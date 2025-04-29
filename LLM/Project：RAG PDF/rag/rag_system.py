import chromadb
import os
from openai import OpenAI
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import streamlit as st
from rag.model_selector import SimpleModelSelector

load_dotenv()

class SimpleRAGSystem:
    def __init__(self, embedding_model="openai", llm_model="openai"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.db = chromadb.PersistentClient(path="./chroma_db")
        self.setup_embedding_function()
        if llm_model == "openai":
            self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.llm = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.collection = self.setup_collection()

    def setup_embedding_function(self):
        try:
            if self.embedding_model == "openai":
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
                )
            elif self.embedding_model == "nomic":
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key="ollama", api_base="http://localhost:11434/v1", model_name="nomic-embed-text"
                )
            else:
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            st.error(f"設定嵌入函式錯誤: {str(e)}")
            raise e

    def setup_collection(self):
        collection_name = f"documents_{self.embedding_model}"
        try:
            try:
                collection = self.db.get_collection(name=collection_name, embedding_function=self.embedding_fn)
                st.info(f"已使用現有的 collection ({self.embedding_model})")
            except:
                collection = self.db.create_collection(name=collection_name, embedding_function=self.embedding_fn, metadata={"model": self.embedding_model})
                st.success(f"新建 collection 成功 ({self.embedding_model})")
            return collection
        except Exception as e:
            st.error(f"建立 collection 出錯: {str(e)}")
            raise e

    def add_documents(self, chunks):
        try:
            if not self.collection:
                self.collection = self.setup_collection()
            self.collection.add(
                ids=[chunk["id"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
            )
            return True
        except Exception as e:
            st.error(f"新增文件時出錯: {str(e)}")
            return False

    def query_documents(self, query, n_results=3):
        try:
            if not self.collection:
                raise ValueError("找不到可用的 collection")
            results = self.collection.query(query_texts=[query], n_results=n_results)
            return results
        except Exception as e:
            st.error(f"查詢文件時出錯: {str(e)}")
            return None

    def generate_response(self, query, context):
        try:
            prompt = f"""
            根據以下內容回答問題。
            如果內容中沒有答案，請直接說不知道。

            內容: {context}

            問題: {query}

            答案:
            """
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini" if self.llm_model == "openai" else "llama3.2",
                messages=[
                    {"role": "system", "content": "你是一位樂於助人的助理。"},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"產生回答時出錯: {str(e)}")
            return None

    def get_embedding_info(self):
        model_selector = SimpleModelSelector()
        model_info = model_selector.embedding_models[self.embedding_model]
        return {"name": model_info["name"], "dimensions": model_info["dimensions"], "model": self.embedding_model}
