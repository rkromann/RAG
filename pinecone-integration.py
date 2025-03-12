
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from dotenv import load_dotenv
# Load .env file
load_dotenv()


from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument, DOCXToDocument
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

# Make sure you have the PINECONE_API_KEY environment variable set
document_store = PineconeDocumentStore(
  index="pinecone-integration",
  metric="cosine",
  dimension=768,
  spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
  )

file_type_router = FileTypeRouter(mime_types=['text/plain','application/pdf','text/markdown', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'])
pdf_converter = PyPDFToDocument()
text_file_converter = TextFileToDocument()
markdown_converter = MarkdownToDocument()
docx_converter = DOCXToDocument()
document_joiner = DocumentJoiner()
document_cleaner = DocumentCleaner()
document_splitter = DocumentSplitter(split_by='word', split_overlap=50)
document_embedder = SentenceTransformersDocumentEmbedder()
document_writer = DocumentWriter(document_store)

indexing = Pipeline()
# indexing.add_component("converter", MarkdownToDocument())
# indexing.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=2))
# indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
# indexing.add_component("writer", DocumentWriter(document_store))


# Adding Componenets
indexing.add_component('file_type_router', file_type_router)
indexing.add_component('text_file_converter', text_file_converter)
indexing.add_component('markdown_converter', markdown_converter)
indexing.add_component('pdf_converter', pdf_converter)
indexing.add_component('docx_converter', docx_converter)
indexing.add_component('document_joiner', document_joiner)
indexing.add_component('document_cleaner', document_cleaner)
indexing.add_component('document_splitter', document_splitter)
indexing.add_component('document_embedder', document_embedder)
indexing.add_component('document_writer', document_writer)

# Connections

indexing.connect('file_type_router.text/plain', 'text_file_converter.sources')
indexing.connect('file_type_router.application/pdf', 'pdf_converter.sources')
indexing.connect('file_type_router.text/markdown', 'markdown_converter.sources')
indexing.connect('file_type_router.application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'docx_converter.sources')
indexing.connect('text_file_converter', 'document_joiner')
indexing.connect('markdown_converter', 'document_joiner')
indexing.connect('pdf_converter', 'document_joiner')
indexing.connect('docx_converter', 'document_joiner')
indexing.connect('document_joiner', 'document_cleaner')
indexing.connect('document_cleaner', 'document_splitter')
indexing.connect('document_splitter', 'document_embedder')
indexing.connect('document_embedder', 'document_writer')





# indexing.connect("converter", "splitter")
# indexing.connect("splitter", "embedder")
# indexing.connect("embedder", "writer")

indexing.run({"file_type_router": {"sources": ["People counting på Århus Universitet.pdf"]}})


from haystack.utils import Secret
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

# Make sure you have the PINECONE_API_KEY environment variable set
document_store = PineconeDocumentStore(
  index="pinecone-integration",
  metric="cosine",
  dimension=768,
  spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
  )
              
prompt_template = """Answer the following query based on the provided context. If the context does
                     not include an answer, reply with 'I don't know'.\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}
                     Answer: 
                  """
import os
query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
query_pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=document_store))
query_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
query_pipeline.add_component("generator", OpenAIGenerator(api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")), model="gpt-4"))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder", "generator")

query = "Which model was the best?"
results = query_pipeline.run(
    {
        "text_embedder": {"text": query},
        "prompt_builder": {"query": query},
    }
)

answer = results["generator"]["replies"][0]
