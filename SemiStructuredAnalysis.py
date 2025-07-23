import Utils
import os
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain_benchmarks import clone_public_dataset, registry
from langchain_benchmarks.rag.tasks.semi_structured_reports import get_file_names

from langchain_benchmarks import registry




# ------------------- 1. Load Online Dataset -------------------
registry = registry.filter(Type="RetrievalTask")
task = registry["Semi-structured Reports"]
clone_public_dataset(task.dataset_id, dataset_name=task.name)
paths = list(get_file_names())
files = [str(p) for p in paths]


# # ------------------- 2. Load & Split -------------------
def load_and_split(files, token_count=500):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=token_count, chunk_overlap=50
    )   
    texts = []
    for file in files:
        loader = PyPDFLoader(file)
        pages = loader.load()
        chunks = splitter.split_documents(pages) # comment out this and the following line if the NoChunk method is intended
        texts.extend([doc.page_content for doc in chunks])
    return texts # return pages if the NoChunk method is intended

texts = load_and_split(files)

# ------------------- 3. Build Retriever -------------------
vectorstore = Chroma.from_texts(texts=texts, embedding=Utils.embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={ "k": 4 })

# ------------------- 4. RAG Chain -------------------
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context, which can include text and tables:
{context}
Question: {question}
""")

chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | prompt
    | Utils.ollama
    | StrOutputParser()
)
print("here2")




# ------------------- 5. Ask Custom Question -------------------
question = "What are the two main resolution methods that the FDIC normally uses for failing banks?"
question_nr = 30

docs = retriever.get_relevant_documents(question)             
retrievals = Utils.format_docs(docs)
answer = chain.invoke(question)


with open(f"SemiStructured/Retrievals/{question_nr}_500.txt", "w") as doc_file:
    doc_file.write(retrievals)

with open(f"SemiStructured/Answers/{question_nr}_500.txt", "w") as answer_file:
    answer_file.write(answer)