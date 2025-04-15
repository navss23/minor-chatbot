import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

# Import from our modules
from config import load_api_key, URLS, CATEGORY_QUESTIONS
from vector_store import get_vectorstore_from_urls
from response import get_response

# Load API key
GOOGLE_API_KEY = load_api_key()
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please check your .env file.")
    st.stop()

# Initialize language model
llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)

# Set page configuration
st.set_page_config(page_title="EDULLM - PERSONALISED EDUCATION PATHWAYS AND CURRICULUM GUIDANCE", page_icon="📚", layout="wide")

# Sidebar content
with st.sidebar:
    st.title("📌 Project Details")
    st.write("🚀 **EduLLM** is a smart education assistant designed to provide **personalized career guidance** and **curriculum recommendations**.")
    st.write("### 👩‍💻 **Developers**")
    st.write("- **Navya Mehta & Aayushi Sharma**")
    st.write("### 🔗 **Useful Links**")
    st.markdown("[📩 Contact Support](mailto:navyamehta.tech@gmail.com)", unsafe_allow_html=True)

# Main page content
st.title("📚 EDULLM - PERSONALISED EDUCATION PATHWAYS AND CURRICULUM GUIDANCE")

st.subheader("🛠️ Technical Overview")
st.write(
    "EduLLM is powered by Google's Gemini AI (gemini-1.5-flash-latest), utilizing LangChain for conversational memory and document retrieval. "
    "It employs retrieval-augmented generation (RAG) with a vector database (ChromaDB) and embeddings from Hugging Face to ensure accurate responses. "
    "The chatbot fetches career-related content from the web via WebBaseLoader, processes it with a context-aware retriever, and provides insightful answers in real-time. "
)

# Initialize vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_urls(URLS)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! I am EduLLM, your personalized education assistant. Ask me anything about career paths, courses, and curriculum guidance! 💡")]

# Career category selection
career_category = st.selectbox("Choose a Career Category:", list(CATEGORY_QUESTIONS.keys()))

if career_category:
    st.subheader(f"Suggested Questions for {career_category}")
    for question in CATEGORY_QUESTIONS[career_category]:
        if st.button(question):
            user_query = question
            response = get_response(user_query, llm, st.session_state.vector_store)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

# Chat interface
if st.session_state.vector_store:
    user_query = st.chat_input("Ask me a question about the content...")
    if user_query:
        response = get_response(user_query, llm, st.session_state.vector_store)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(f"""{message.content}""")
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(f"""{message.content}""")
