import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from langchain.vectorstores.chroma import Chroma
from langchain.schema.output_parser import StrOutputParser
import os
import Utils



_chunkNumber = 7



configuration = "SemanticChunk{}_".format(_chunkNumber)

chroma_configuration = "SemanticChunk"
chroma_persist_path = f"/home/kristian/chromadb/{chroma_configuration}"



#-------------------------- 1. Load ----------------------------------
docs = Utils.load_langchain_docs()
print("here1")

#--------------------- 2. Split & Embed ------------------------------
all_splits = Utils.semantic_split_docs(docs)




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



#----------------------3. Retriever ----------------------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={ "k": _chunkNumber })
print("here3")

response_generator = (Utils.prompt | Utils.ollama | StrOutputParser()).with_config(
    run_name="GenerateResponse",
)


# ---------------------- Generate Answer ----------------------
question = "how to run a runnable"
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


with open(f"{answer_path}/{configuration}.txt", "w") as answer_file:
    answer_file.write(formatted_context +  "\n\n\n\nAnswer: " + chain_output)
