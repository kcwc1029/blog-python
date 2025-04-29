import streamlit as st
from rag.model_selector import SimpleModelSelector
from rag.pdf_processor import SimplePDFProcessor
from rag.rag_system import SimpleRAGSystem

def main():
    st.title("🤖 簡單版 RAG 系統")

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    model_selector = SimpleModelSelector()
    llm_model, embedding_model = model_selector.select_models()

    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None
        st.warning("嵌入模型已變更，請重新上傳文件")

    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)

        embedding_info = st.session_state.rag_system.get_embedding_info()
        st.sidebar.info(
            f"目前使用模型:\n"
            f"- 名稱: {embedding_info['name']}\n"
            f"- 維度: {embedding_info['dimensions']}"
        )
    except Exception as e:
        st.error(f"初始化 RAG 系統時出錯: {str(e)}")
        return

    pdf_file = st.file_uploader("上傳 PDF 文件", type="pdf")

    if pdf_file and pdf_file.name not in st.session_state.processed_files:
        processor = SimplePDFProcessor()
        with st.spinner("處理 PDF 中..."):
            try:
                text = processor.read_pdf(pdf_file)
                chunks = processor.create_chunks(text, pdf_file)
                if st.session_state.rag_system.add_documents(chunks):
                    st.session_state.processed_files.add(pdf_file.name)
                    st.success(f"成功處理 {pdf_file.name}")
            except Exception as e:
                st.error(f"處理 PDF 時出錯: {str(e)}")

    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("🔍 問問題")
        query = st.text_input("輸入問題:")

        if query:
            with st.spinner("生成回答中..."):
                results = st.session_state.rag_system.query_documents(query)
                if results and results["documents"]:
                    response = st.session_state.rag_system.generate_response(
                        query, results["documents"][0]
                    )

                    if response:
                        st.markdown("### 📝 回答:")
                        st.write(response)

                        with st.expander("查看來源段落"):
                            for idx, doc in enumerate(results["documents"][0], 1):
                                st.markdown(f"**段落 {idx}:**")
                                st.info(doc)
    else:
        st.info("👆 請先上傳 PDF 文件")

if __name__ == "__main__":
    main()