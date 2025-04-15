from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval_qa.base import RetrievalQA

def get_context_retriever_chain(llm, vector_store):
    """
    Create a context-aware retriever chain.

    Args:
        llm: Language model
        vector_store: Vector store for retrieving documents

    Returns:
        Chain: Context-aware retriever chain
    """
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to retrieve relevant information.")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(llm, vector_store):
    """
    Create a conversational RAG chain.

    Args:
        llm: Language model
        vector_store: Vector store for retrieving documents

    Returns:
        Chain: Conversational RAG chain
    """
    retriever = vector_store.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
