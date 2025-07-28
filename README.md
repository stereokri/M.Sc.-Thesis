# M.Sc.-Thesis


This Repository shows all the results of the Master Thesis titled "Inspection and Evaluation of RAG Methods". It is structured as follows:

## Results

This directory contains the manually inspected results used for evaluating the RAG pipelines.

- Each **dataset** has its own subdirectory.
- Within each dataset, every **query** is placed in a separate folder, named after the original query.
- Inside each query folder, you’ll find **text files** named according to the method used.

For example:

HyDE_500_100_7.txt

This filename indicates that the **HyDE** method was applied using:
- Chunk size: `500`
- Chunk overlap: `100`
- Number of retrieved chunks: `7`

Each such file contains the retrieved context and the generated answer for that specific query and method configuration.

## KnowledgeBase

This directory contains all document sets used as the supporting evidence/ground truth for the tested RAG systems. In a separate **AdditionalDocs_QA** folder, one can find all additional text documents injected into the knowledge base with the purpose of further inspection, as discussed in the thesis.

## RAGScripts

This is the directory that contains all scripts used for testing. Each script denotes a specific variant of a RAG pipeline, as discussed in the thesis. To tune the hyperparameters it is necessary to change the following values at the top of each `.py` file:

- `_chunkNumber`
- `_chunkSize`
- `_chunkOverlap`

Note that depending on the method employed, it could be the case that not all three hyperparamters are available to tune. E.g., the Semantic Chunking method only needs the `_chunkNumber` method.

In order to successfully run all scripts, it is needed to set the following private API keys in the Utils.py file: LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY, and set the TOKENIZERS_PARALLELISM flag to false.


In order to run experiments as a batch with the results forwared to the web-based dashboard supported by LangSmith it is needed to add the following snippet to the end of the script intended to execute:

<pre> 
from langchain_benchmarks.rag import get_eval_config
from langsmith.client import Client
import uuid

retriever_factory = langchain_docs.retriever_factories["basic"]
retriever = retriever_factory(
    Utils.embeddings,
    transform_docs=all_splits,
    transformation_name="recursive split",
    search_kwargs={"k": _chunkNumber},
)
chain_factory = langchain_docs.architecture_factories["conversational-retrieval-qa"]

  
client = Client()
RAG_EVALUATION = get_eval_config()
run_uid = uuid.uuid4().hex[:6]


chunked_results = client.run_on_dataset(
    dataset_name=langchain_docs.name,
    llm_or_chain_factory=partial(chain_factory, <retriever_name>, llm=ollama),
    evaluation=RAG_EVALUATION, 
    project_name=f"HyDE chunked 500 100 7 neighbours {run_uid}",
    project_metadata={
        "index_method": "basic",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "embedding_model": "thenlper/gte-base",
        "llm": "llama3:70b-instruct-q2_K",
    },
    verbose=True,
)


  
</pre>
