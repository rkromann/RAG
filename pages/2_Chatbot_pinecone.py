
from itertools import chain
from typing import Any, List

from haystack.components.converters import PyPDFToDocument, MarkdownToDocument, TextFileToDocument, OutputAdapter, DOCXToDocument
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.core.component.types import Variadic

from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack_integrations.components.generators.cohere import CohereChatGenerator, CohereGenerator


from haystack.dataclasses import ChatMessage
from haystack import Pipeline
from haystack import component

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access the API key
os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')

# Make sure you have the PINECONE_API_KEY environment variable set
document_store = PineconeDocumentStore(
        index='seven-wonders',
        namespace="default",
        dimension=384,
    metric="cosine",
    spec={"serverless": {"region": "us-east-1", "cloud": "aws"}}
)
# file_type_router = FileTypeRouter(mime_types=['text/plain','application/pdf','text/markdown', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'])
# pdf_converter = PyPDFToDocument()
# text_file_converter = TextFileToDocument()
# markdown_converter = MarkdownToDocument()
# docx_converter = DOCXToDocument()
# document_joiner = DocumentJoiner()
# document_cleaner = DocumentCleaner()
# document_splitter = DocumentSplitter(split_by='word', split_overlap=50)
# document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L12-v2")
# document_writer = DocumentWriter(document_store)





@component
class ListJoiner:
  def __init__(self, _type: Any):
    component.set_output_types(self, values=_type)

  def run(self, values:Variadic[Any]):
    result = list(chain(*values))
    return {'values':result}
  

memory_store = InMemoryChatMessageStore()

query_rephrase_template="""
        Rewrite the question for search while keeping its meaning and key terms intact.
        If the conversation history is empty, DO NOT change the query.
        Use conversation history only if necessary, and avoid extending the query with your own knowledge.
        If no changes are needed, output the current question as is.

        Conversation history:
        {% for memory in memories %}
            {{ memory.content }}
        {% endfor %}

        User Query: {{query}}
        Rewritten Query:
"""


conversational_rag = Pipeline()

#Query rephrasing components
conversational_rag.add_component("query_rephrase_prompt_builder",PromptBuilder(query_rephrase_template))
conversational_rag.add_component('query_rephrase_llm',CohereGenerator())
conversational_rag.add_component('list_to_str_adapter', OutputAdapter(template="{{ replies[0] }}", output_type=List[float]))

#RAG components
conversational_rag.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
conversational_rag.add_component('retriever', PineconeEmbeddingRetriever(document_store=document_store, top_k=3))
conversational_rag.add_component('prompt_builder', ChatPromptBuilder(variables=["query", "documents", "memories"],required_variables=['query', 'documents', 'memories']))
conversational_rag.add_component('llm', CohereChatGenerator())

#Memory components
conversational_rag.add_component('memory_retriever',ChatMessageRetriever(memory_store))
conversational_rag.add_component('memory_writer', ChatMessageWriter(memory_store))
conversational_rag.add_component('memory_joiner', ListJoiner(List[ChatMessage]))


#Query Rephrasing Connections
conversational_rag.connect('memory_retriever', 'query_rephrase_prompt_builder.memories')
conversational_rag.connect('query_rephrase_prompt_builder.prompt', 'query_rephrase_llm' )
conversational_rag.connect('query_rephrase_llm.replies', 'list_to_str_adapter')
# conversational_rag.connect('list_to_str_adapter', 'retriever.query_embedding')

#RAG connections
conversational_rag.connect('text_embedder.embedding', 'retriever.query_embedding')
conversational_rag.connect('retriever.documents', 'prompt_builder.documents')
conversational_rag.connect('prompt_builder.prompt', 'llm.messages')
conversational_rag.connect('llm.replies', 'memory_joiner')

#Memory Connections
conversational_rag.connect('memory_joiner','memory_writer')
conversational_rag.connect('memory_retriever','prompt_builder.memories')


system_message = ChatMessage.from_system("""You are an intelligent and cheerful AI assistant specialized in assisting humans with queries based on provided supporting documents and conversation history. 
                                         Always prioritize accurate and concise answers derived from the documents, and offer contextually relevant follow-up questions to maintain an engaging and helpful conversation. 
                                         If the answer is not present in the documents, politely inform the user while suggesting alternative ways to help""")

user_message_template ="""Based on the conversation history and the provided supporting documents, provide a brief and accurate answer to the question.
                          Make the conversation feel more natural and engaging

- Format your response for clarity and readability, using bullet points, paragraphs, or lists where necessary.
- Note: Supporting documents are not part of the conversation history.
- If the question cannot be answered using the supporting documents, respond with: "The answer is not available in the provided documents."

Conversation History:
{% for memory in memories %}
{{ memory.content }}
{% endfor %}

Supporting Documents:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ query }}
Answer:

"""
user_message = ChatMessage.from_user(user_message_template)
messages = [system_message, user_message]
question = "When was the Colossum of Rhodes built?"
res = conversational_rag.run(
    data = {'query_rephrase_prompt_builder' : {'query': question},
            'prompt_builder': {'template': messages, 'query': question},
            'memory_joiner': {'values': [ChatMessage.from_user(question)]}},
    include_outputs_from=['llm','query_rephrase_llm'])

bot_message = res['llm']['replies'][0].text