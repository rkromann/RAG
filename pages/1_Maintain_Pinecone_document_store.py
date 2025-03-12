
from haystack.components.converters import PyPDFToDocument, MarkdownToDocument, TextFileToDocument, DOCXToDocument
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack import Pipeline
import streamlit as st
import pandas as pd
import os, pathlib
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
load_dotenv()

container1 = st.container(border=True)
container2 = st.container(border=True)

def count_documents(index):
	document_store = PineconeDocumentStore(index=index)
	return document_store.count_documents()

def show_indexes():
	pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
	index_list = pc.list_indexes()
	index_names = [item['name'] for item in index_list]
	index_dimensions = [item['dimension'] for item in index_list]
	index_documents = [count_documents(index) for index in index_names]
	df = pd.DataFrame.from_dict({'Index name': index_names, 'Dimension': index_dimensions, 'Number of chunks': index_documents})
	container2.write("Indexes available in your Pinecone instance")
	container2.write(df)

with container1:
	st.write("Click button to see which indexes are available in your Pinecone instance")
	st.button("Refresh indexes", on_click=show_indexes)

with st.form("inputform", clear_on_submit=True, enter_to_submit=False):
	index_name = st.text_input("Enter the name of one of the above indexes to add documents to or create a new one")
	files = st.file_uploader("Upload files to be added to the index", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True)
	model = st.selectbox("Select a model to embed the documents", ["all-MiniLM-L12-v2", "multilingual-e5-large-instruct"])
	model_dict = {"all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2", "multilingual-e5-large-instruct": "intfloat/multilingual-e5-large-instruct"}
	model_dimension_dict = {"all-MiniLM-L12-v2": 384, "multilingual-e5-large-instruct": 1024}
	model_dimension = model_dimension_dict[model]
	model = model_dict[model]
	submit_button = st.form_submit_button("Submit")

if files:

	os.chdir(pathlib.Path(__file__).parent.parent.joinpath('sources'))

	for file in files:
		with open(file.name, 'wb') as handle:
			handle.write(file.getvalue())

	files = os.listdir()

	# Make sure you have the PINECONE_API_KEY environment variable set
	document_store = PineconeDocumentStore(
			index=index_name,
			namespace="default",
			dimension=model_dimension,
		metric="cosine",
		spec={"serverless": {"region": "us-east-1", "cloud": "aws"}}
	)

	file_type_router = FileTypeRouter(mime_types=['text/plain','application/pdf','text/markdown', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'])
	pdf_converter = PyPDFToDocument()
	text_file_converter = TextFileToDocument()
	markdown_converter = MarkdownToDocument()
	docx_converter = DOCXToDocument()
	document_joiner = DocumentJoiner()
	document_cleaner = DocumentCleaner()
	document_splitter = DocumentSplitter(split_by='word', split_overlap=50)
	document_embedder = SentenceTransformersDocumentEmbedder(model = model)
	document_writer = DocumentWriter(document_store)

	preprocessing_pipeline = Pipeline()

	# Adding Componenets
	preprocessing_pipeline.add_component('file_type_router', file_type_router)
	preprocessing_pipeline.add_component('text_file_converter', text_file_converter)
	preprocessing_pipeline.add_component('markdown_converter', markdown_converter)
	preprocessing_pipeline.add_component('pdf_converter', pdf_converter)
	preprocessing_pipeline.add_component('docx_converter', docx_converter)
	preprocessing_pipeline.add_component('document_joiner', document_joiner)
	preprocessing_pipeline.add_component('document_cleaner', document_cleaner)
	preprocessing_pipeline.add_component('document_splitter', document_splitter)
	preprocessing_pipeline.add_component('document_embedder', document_embedder)
	preprocessing_pipeline.add_component('document_writer', document_writer)

	# Connections

	preprocessing_pipeline.connect('file_type_router.text/plain', 'text_file_converter.sources')
	preprocessing_pipeline.connect('file_type_router.application/pdf', 'pdf_converter.sources')
	preprocessing_pipeline.connect('file_type_router.text/markdown', 'markdown_converter.sources')
	preprocessing_pipeline.connect('file_type_router.application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'docx_converter.sources')
	preprocessing_pipeline.connect('text_file_converter', 'document_joiner')
	preprocessing_pipeline.connect('markdown_converter', 'document_joiner')
	preprocessing_pipeline.connect('pdf_converter', 'document_joiner')
	preprocessing_pipeline.connect('docx_converter', 'document_joiner')
	preprocessing_pipeline.connect('document_joiner', 'document_cleaner')
	preprocessing_pipeline.connect('document_cleaner', 'document_splitter')
	preprocessing_pipeline.connect('document_splitter', 'document_embedder')
	preprocessing_pipeline.connect('document_embedder', 'document_writer')

	a = preprocessing_pipeline.run({'file_type_router': {'sources': files}})['document_writer']['documents_written']
	st.write(f"Uploaded {len(files)} files in {a} chunks to the index {index_name}")

	for file in files:
		os.remove(file)