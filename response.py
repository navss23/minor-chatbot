import streamlit as st
from retriever import get_conversational_rag_chain

def get_response(user_input, llm, vector_store):
    """
    Generate a response for the user input.

    Args:
        user_input (str): User's query
        llm: Language model
        vector_store: Vector store for retrieving documents

    Returns:
        str: Generated response
    """
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(user_input)

    if docs:
        conversation_rag_chain = get_conversational_rag_chain(llm, vector_store)
        response = conversation_rag_chain.invoke({"query": user_input})
        return response.get("result", "I'm sorry, I couldn't generate a response.")
    else:
        fallback_prompt = f"""
        You are EduLLM, an education-focused assistant. ONLY answer questions related to education, careers, or curriculum.
        If the question is unrelated, politely decline to answer.

        User question: {user_input}
        """
        return llm.invoke(fallback_prompt).content
