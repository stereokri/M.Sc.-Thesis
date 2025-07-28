import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from langchain.vectorstores.chroma import Chroma
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.prompts import ChatPromptTemplate
import os
import Utils
from langchain.storage import InMemoryStore
import hashlib

_chunkNumber = 7
_chunkSize = 1000
_chunkOverlap = 200

configuration = "ParentDoc{}_{}_{}".format(_chunkSize, _chunkOverlap, _chunkNumber)

chroma_configuration = "SimpleChunk{}_{}".format(_chunkSize, _chunkOverlap)
chroma_persist_path = f"/home/kristian/chromadb/{chroma_configuration}"


#-------------------------- 1. Load ----------------------------------
docs = Utils.load_langchain_docs()
print("here1")

#--------------------- 2. Split & Embed ------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_chunkSize, chunk_overlap=_chunkOverlap, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print("here1.5")


if os.path.exists(chroma_persist_path):
    vectorstore = Chroma(
        collection_name="langchain",
        embedding_function=Utils.embeddings,
        persist_directory=chroma_persist_path,
    )
else:
    # First‐time build: embed and persist
    os.makedirs(chroma_persist_path, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=Utils.embeddings,
        persist_directory=chroma_persist_path,
    )
    vectorstore.persist()
print("here2")


#----------------------3. Retriever ----------------------------------
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    search_kwargs = {"k": _chunkNumber},
    docstore=store,
    child_splitter=Utils.text_splitter,
)
retriever.add_documents(docs, ids=None)
print("here3")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant answering questions about LangChain."
            "\nHere are the retrieved documents:\n{context}\n"
            "Only answer solely based on the documents with AT MOST 3 sentences!"
            # "If it is a yes or no question only respond with yes or no.",
        ),
        ("human", "{question}"),
    ]
)

response_generator = (prompt | Utils.ollama | StrOutputParser()).with_config(
    run_name="GenerateResponse",
)

print("here5")

# ---------------------- Generate Answer ----------------------
question = "how do I run gpt-4 on anthropic?"
answer_path = f"/home/kristian/QA/{question}"
os.makedirs(answer_path, exist_ok=True)


retrieved_docs = retriever.invoke(question)
formatted_context = Utils.format_docs(retrieved_docs)

print("Retrieved Docs:\n")
print(formatted_context)


chain_output = response_generator.invoke({
    "context": formatted_context,
    "question": question
})

print("Answer:\n")
print(chain_output)


with open(f"/home/kristian/QA/{question}/{configuration}.txt", "w") as answer_file:
    answer_file.write(formatted_context +  "\n\n\n\nAnswer: " + chain_output)
