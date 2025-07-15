import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from langchain.vectorstores.chroma import Chroma
from langchain.schema.output_parser import StrOutputParser
import os
import Utils
import hashlib

_chunkNumber = 7
_chunkSize = 1000
_chunkOverlap = 200

configuration = "SimpleChunk{}_{}_{}".format(_chunkSize, _chunkOverlap, _chunkNumber)
chroma_configuration = "SimpleChunk{}_{}".format(_chunkSize, _chunkOverlap)
chroma_persist_path = f"/home/kristian/chromadb/{chroma_configuration}"


#-------------------------- 1. Load ----------------------------------
docs = Utils.load_langchain_docs()

#-------------------------- 2. Split ---------------------------------
all_splits = Utils.split_docs(docs, _chunkSize, _chunkOverlap)


#-------------------------- 3. Embed ---------------------------------
if os.path.exists(chroma_persist_path):
     # Look up persisted directory
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


#----------------------4. Retrieve ----------------------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={ "k": _chunkNumber })


#--------------------- 5. Generate -----------------------------------
response_generator = (Utils.prompt | Utils.ollama | StrOutputParser()).with_config(
    run_name="GenerateResponse",
)

# ---------------------- Generate Answer ----------------------
question = "how to load CSV files???"

answer_path = f"/home/kristian/QA/{question}"
os.makedirs(answer_path, exist_ok=True)


formatted_context = Utils.format_docs(retriever.invoke(question))

chain_output = response_generator.invoke({
    "context": formatted_context,
    "question": question
})


with open(f"{answer_path}/{configuration}.txt", "w") as answer_file:
    answer_file.write(formatted_context +  "\n\n\n\nAnswer: " + chain_output)