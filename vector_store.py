import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_vectorstore_from_urls(urls):
    """
    Load documents from URLs, split into chunks, and create a vector store.

    Args:
        urls (list): List of URLs to load documents from

    Returns:
        Chroma: Vector store containing document chunks
    """
    all_chunks = []
    try:
        for url in urls:
            loader = WebBaseLoader(url, header_template={"User-Agent": "Mozilla/5.0"})
            document = loader.load()
            text_splitter = RecursiveCharacterTextSplitter()
            document_chunks = text_splitter.split_documents(document)
            all_chunks.extend(document_chunks)

        return Chroma.from_documents(
            all_chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
    except Exception as e:
        st.error(f"Error loading websites: {e}")
        return None
