import os
from typing import Sequence
from langchain.schema.document import Document
from langchain_community.llms import Ollama
from langchain_benchmarks import registry, clone_public_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker


embeddings = HuggingFaceEmbeddings(
    model_name="thenlper/gte-base",
    model_kwargs={"device": 0},  # Comment out to use CPU
)


''''
@param: a sequence of document chunks
return: XML-formatted string to contain all retrieved chunks separated by newlines 
each chunk shows: retrieval rank, document source, text content 
'''
def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = (
            f"<document index='{i}'>\n"
            f"<source>{doc.metadata.get('source')}</source>\n"
            f"<doc_content>{doc.page_content}</doc_content>\n"
            "</document>\n\n\n\n\n"
        )
        formatted_docs.append(doc_string)
    formatted_str = "\n".join(formatted_docs)
    return f"<documents>\n{formatted_str}\n</documents>"


def split_docs(docs, chunkSz, chunkOl):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunkSz, chunk_overlap=chunkOl, add_start_index=True
    )
    return text_splitter.split_documents(docs)


def semantic_split_docs(docs):
    text_splitter = SemanticChunker(embeddings = embeddings)
    return text_splitter.split_documents(docs)


def format_docs_metadata(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = (
            f"<document index='{i}'>\n"
            f"<source>{doc.metadata.get('source')}</source>\n"
            f"<doc_title>{doc.metadata.get('title')}</doc_title>\n"
            "</document>\n\n\n\n\n"
        )
        formatted_docs.append(doc_string)
    formatted_str = "\n".join(formatted_docs)
    return f"<documents>\n{formatted_str}\n</documents>"

''''
Commented out lines needed to add further documents to the knowledge base
'''
def load_langchain_docs():
    registry.filter(Type="RetrievalTask")
    langchain_docs = registry["LangChain Docs Q&A"]
    clone_public_dataset(langchain_docs.dataset_id, dataset_name=langchain_docs.name)
    docs = list(langchain_docs.get_docs())
    # docs = list()
    # loader = TextLoader("/home/kristian/BenchMarkingPythonScripts/AdditionalDocs/Anthropic.txt", encoding="utf-8")
    # local_doc = loader.load()
    # docs = docs + local_doc
    return docs


MODEL = 'llama3:70b-instruct-q2_K'


ollama = Ollama(
    base_url='http://localhost:11434',
    model=MODEL,
    temperature=0,   
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
)


embeddings = HuggingFaceEmbeddings(
    model_name="thenlper/gte-base",
    model_kwargs={"device": 0},  # Comment out to use CPU
)

prompt = ChatPromptTemplate.from_messages(
    [("system",
    "You are an AI assistant answering questions about LangChain."
    "\nHere are the retrieved documents:\n{context}\n"
    "Only answer solely based on the documents with AT MOST 3 sentences!"),
    ("human", "{question}")]
)

def generate_hypothetical_answer(question: str) -> str:
    hyde_prompt = f"Answer this question as best as you can even if you must hallucinate: {question} \
    Do not include your thought process or any internal reasoning:\n\n"
    return ollama.invoke(hyde_prompt)


def rephrase_question(question: str) -> str:
    rephrase_prompt = f"You are an assistant tasked with taking a natural language \
    query from a user and rephrase it if it is wrongly formulated \
    or not appropriate for the retrieval task. Here is the user query: {question}"
    return ollama.invoke(rephrase_prompt)


def split_multiple_questions(question: str) -> str:
    split_prompt = f"You are tasked only with transforming the input query you receive.\
    If the query contains several questions joined by “and” or “or”, and all follow the same question pattern\
    (e.g., “What is A and B and C and D?”), split them into distinct questions.\
    Example:\
    Input: How to make A and B?” Output: “How to make A?+++How to make B?”\
    Do not paraphrase, reword, or explain anything.\
    If the above condition is not met, return the original query exactly as received.\
    Only return the final output. No reasoning or explanation. The input question is: {question}"
    
    
    return ollama.invoke(split_prompt)


def split_questions(text):
    # Split by '+++' 
    questions = [q for q in text.split('+++')]
    return questions