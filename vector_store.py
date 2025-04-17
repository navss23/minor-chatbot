import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_vectorstore_from_urls(urls, show_progress=True):
    """
    Load documents from URLs, split into chunks, and create a vector store.

    Args:
        urls (list): List of URLs to load documents from
        show_progress (bool): Whether to show a progress bar

    Returns:
        Chroma: Vector store containing document chunks
    """
    all_chunks = []
    failed_urls = []

    # Skip if no URLs provided
    if not urls:
        st.warning("No scrapable URLs available.")
        return None

    # Create a progress bar if requested
    if show_progress:
        progress_bar = st.progress(0)
        progress_text = st.empty()

    try:
        for i, url in enumerate(urls):
            if show_progress:
                progress_text.text(f"Processing URL {i+1}/{len(urls)}: {url}")
                progress_bar.progress((i) / len(urls))

            try:
                loader = WebBaseLoader(url, header_template={"User-Agent": "Mozilla/5.0"})
                document = loader.load()
                text_splitter = RecursiveCharacterTextSplitter()
                document_chunks = text_splitter.split_documents(document)
                all_chunks.extend(document_chunks)
            except Exception as e:
                failed_urls.append((url, str(e)))
                st.warning(f"Failed to load {url}: {e}")
                continue

        if show_progress:
            progress_bar.progress(1.0)
            progress_text.text("Processing complete!")

        # Show summary of failures
        if failed_urls:
            st.warning(f"Failed to load {len(failed_urls)} URLs out of {len(urls)}")

        # Return None if no documents were loaded
        if not all_chunks:
            st.error("No documents were successfully loaded. Please check your URLs.")
            return None

        return Chroma.from_documents(
            all_chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
    except Exception as e:
        st.error(f"Error in vector store creation: {e}")
        return None
    finally:
        # Clear progress elements
        if show_progress:
            progress_bar.empty()
            progress_text.empty()

def get_category_vectorstore(urls_dict, category=None):
    """
    Load documents from specific category URLs or all URLs.

    Args:
        urls_dict (dict): Dictionary of URLs organized by category
        category (str, optional): Specific category to load. If None, load all URLs.

    Returns:
        Chroma: Vector store containing document chunks
    """
    if category and category in urls_dict:
        urls_to_load = urls_dict[category]
        if not urls_to_load:
            st.warning(f"No scrapable URLs available for category: {category}")
            return None
    else:
        # Flatten all URLs from all categories
        urls_to_load = [url for category_urls in urls_dict.values() for url in category_urls]
        if not urls_to_load:
            st.warning("No scrapable URLs available.")
            return None

    return get_vectorstore_from_urls(urls_to_load)
