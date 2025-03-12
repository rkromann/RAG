import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to RAG system demo")

st.sidebar.success("Select a function above.")

st.markdown(
"""
**Manintain Pinecone document store** can show the contents of your Pinecone instance and add new text documents, 
both to an existing index and to a new index
"""
)