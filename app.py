import os
import tempfile
import streamlit as st

from rag_core import ask_rag, index_pdf_file, index_folder

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")

st.title("📄 RAG Chatbot for Papers")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

with st.sidebar:
    st.header("Indexing")

    if st.button("Re-index all PDFs in /papers"):
        try:
            stored = index_folder("papers", reset=True)
            st.success(f"Klart. {stored} chunks indexerades.")
        except Exception as e:
            st.error(str(e))

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Index uploaded PDFs"):
            total = 0
            indexed_names = []

            for uploaded in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                try:
                    count = index_pdf_file(tmp_path, source_name=uploaded.name)
                    total += count
                    indexed_names.append(uploaded.name)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            st.session_state.indexed_files.extend(indexed_names)
            st.success(f"Klart. {total} chunks indexerades från uploadade filer.")

    st.markdown("---")
    st.subheader("Tips")
    st.write("Fråga till exempel:")
    st.code("What is PRCO?")
    st.code("What does ScholScan say about RAG?")
    st.code("What is D2Skill?")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Indexed uploaded files")
    if st.session_state.indexed_files:
        for name in st.session_state.indexed_files:
            st.write("-", name)
    else:
        st.write("Inga uploadade filer indexerade ännu.")

with col1:
    st.subheader("Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                if msg["sources"]:
                    st.markdown("**Sources:**")
                    for s in msg["sources"]:
                        st.write("-", s)

    user_input = st.chat_input("Ask something about your papers...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        history_for_rag = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            if m["role"] in ["user", "assistant"]
        ][:-1]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_rag(user_input, history_for_rag)

            st.markdown(result["answer"])

            if result["sources"]:
                st.markdown("**Sources:**")
                for s in result["sources"]:
                    st.write("-", s)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )