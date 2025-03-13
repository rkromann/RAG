import streamlit as st
import torch

st.set_page_config(page_title="Hello", page_icon="foaroedweb.png")

st.write("# Welcome to RAG system demo")

st.sidebar.success("Select a function above.")

st.markdown(
"""
**Manintain Pinecone document store** can show the contents of your Pinecone instance and add new text documents, 
both to an existing index and to a new index
"""
)

torch.classes.__path__ = [] # add this line to manually set it to empty. 