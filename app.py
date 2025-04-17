import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

# Import from our modules
from config import load_api_key, URLS, CATEGORY_QUESTIONS
from vector_store import get_vectorstore_from_urls, get_category_vectorstore
from response import get_response
from url_utils import test_category_urls, get_all_scrapable_urls

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
    st.write("🚀 **EduLLM** is a smart education assistant designed to provide **personalized career guidance** and **curriculum recommendations** for Indian students.")
    st.write("### 👩‍💻 **Developers**")
    st.write("- **Navya Mehta & Aayushi Sharma**")
    st.write("### 🔗 **Useful Links**")
    st.markdown("[📩 Contact Support](mailto:navyamehta.tech@gmail.com)", unsafe_allow_html=True)

    # Add option to test URLs
    st.subheader("Knowledge Base Settings")

    if st.button("Test URLs for Scraping"):
        st.session_state.tested_urls = test_category_urls(URLS)

    # Only show these options if URLs have been tested
    if "tested_urls" in st.session_state:
        load_all = st.checkbox("Load all scrapable URLs (may take longer)", value=False)

        if not load_all:
            available_categories = [cat for cat, urls in st.session_state.tested_urls.items() if urls]
            if available_categories:
                selected_category = st.selectbox(
                    "Choose specific category to load:",
                    available_categories
                )
            else:
                st.error("No categories have scrapable URLs.")
                selected_category = None

        if st.button("Load Knowledge Base"):
            with st.spinner("Loading knowledge base... This may take a minute..."):
                if load_all:
                    all_urls = get_all_scrapable_urls(st.session_state.tested_urls)
                    if all_urls:
                        st.session_state.vector_store = get_vectorstore_from_urls(all_urls)
                        if st.session_state.vector_store:
                            st.success("All categories loaded successfully!")
                    else:
                        st.error("No scrapable URLs available.")
                else:
                    if selected_category:
                        st.session_state.vector_store = get_category_vectorstore(
                            st.session_state.tested_urls, selected_category
                        )
                        if st.session_state.vector_store:
                            st.success(f"Knowledge base for {selected_category} loaded successfully!")

# Main page content
st.title("📚 EDULLM - PERSONALISED EDUCATION PATHWAYS AND CURRICULUM GUIDANCE")

st.subheader("🛠️ Technical Overview")
st.write(
    "EduLLM is powered by Google's Gemini AI (gemini-1.5-flash-latest), utilizing LangChain for conversational memory and document retrieval. "
    "It employs retrieval-augmented generation (RAG) with a vector database (ChromaDB) and embeddings from Hugging Face to ensure accurate responses. "
    "The chatbot fetches India-specific career-related content from the web via WebBaseLoader, processes it with a context-aware retriever, and provides insightful answers in real-time. "
)

# Initialize warning message
if "tested_urls" not in st.session_state:
    st.info("Please test URLs from the sidebar before loading the knowledge base.")
elif "vector_store" not in st.session_state:
    st.warning("Please load a knowledge base from the sidebar to start chatting.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! I am EduLLM, your personalized education assistant for Indian students. Ask me anything about career paths, courses, and curriculum guidance in India! 💡")]

# Career category selection
career_category = st.selectbox("Choose a Career Category for Questions:", list(CATEGORY_QUESTIONS.keys()))

if career_category:
    st.subheader(f"Suggested Questions for {career_category}")
    cols = st.columns(2)  # Display questions in two columns

    for i, question in enumerate(CATEGORY_QUESTIONS[career_category]):
        col = cols[i % 2]
        with col:
            if st.button(question, key=f"q_{i}"):
                user_query = question
                if "vector_store" in st.session_state and st.session_state.vector_store:
                    with st.spinner("Generating response..."):
                        response = get_response(user_query, llm, st.session_state.vector_store)
                        st.session_state.chat_history.append(HumanMessage(content=user_query))
                        st.session_state.chat_history.append(AIMessage(content=response))
                else:
                    st.warning("Please load a knowledge base first.")

# Chat interface without audio
if "vector_store" in st.session_state and st.session_state.vector_store:
    user_query = st.chat_input("Ask me a question about education and careers in India...")

    if user_query:
        with st.spinner("Generating response..."):
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
