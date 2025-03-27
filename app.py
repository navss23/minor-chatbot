# Install required libraries before running:
# pip install streamlit langchain beautifulsoup4 python-dotenv chromadb google-generativeai sentence-transformers

import google.generativeai as genai
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(env_path)

# Verify if the Google API key is loaded
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please check your .env file.")
    st.stop()

# ✅ Use LangChain's GoogleGenerativeAI wrapper
llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)

# ✅ Hardcoded URL (Replace with your desired website)
FIXED_URL = "https://medium.com/study-guide/career-paths-after-10th-and-12th-standard-30f069541c48"

def get_vectorstore_from_url(url):
    """Fetches website data, splits it into chunks, and creates a vectorstore."""
    try:
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        return Chroma.from_documents(
            document_chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
    except Exception as e:
        st.error(f"Error loading website: {e}")
        return None

def get_context_retriever_chain(vector_store):
    """Creates a retriever chain for fetching relevant context."""
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to retrieve relevant information.")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(retriever_chain):
    """Creates a retrieval-augmented generation (RAG) chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    """Generates a response using Google Gemini."""
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = llm.invoke(user_input)
    return response if response else "I'm sorry, I couldn't generate a response."

# App configuration
st.set_page_config(page_title="PERSONALISED EDUCATION PATHWAYS AND CURRICULUM GUIDANCE", page_icon="📚", layout="wide")

# ✅ Sidebar for display only
with st.sidebar:
    st.title("📌 Project Details")
    st.write("This chatbot provides personalized education pathways and curriculum guidance.")
    st.write("### Developer:")
    st.write("- **Name:** Navya Mehta & Aayushi Sharma")  # Replace with your name

    st.write("### Useful Links:")
    st.markdown("[Visit Source Website](https://medium.com/study-guide/career-paths-after-10th-and-12th-standard-30f069541c48)", unsafe_allow_html=True)

    st.markdown("[Contact Support](mailto:navyamehta.tech@gmail.com)", unsafe_allow_html=True)  # Replace with actual email

st.title("📚 PERSONALIZED EDUCATION PATHWAYS AND CURRICULUM GUIDANCE")

st.write(
    "EduMentor leverages the cutting-edge RAG (Retrieval-Augmented Generation) function to provide in-depth, contextually rich answers to complex educational queries. "
    "This AI-driven approach combines extensive knowledge retrieval with dynamic response generation, offering students a deeper, more nuanced understanding of their career options and fostering a more interactive, exploratory learning environment."
)



# ✅ Load vector store once at the start
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url(FIXED_URL)

if st.session_state.vector_store:
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I assist you?")]

    user_query = st.chat_input("Ask me a question about the content...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display chat messages
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(f"""{message.content}""")
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(f"""{message.content}""")
