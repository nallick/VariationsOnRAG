# to start service: .venv/bin/fastapi run main.py

import fastapi

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv("../.env")

from postgres_retrieval_service_types import *
from vector_database_PGVector import *


def _load_embedding_function() -> HuggingFaceEmbeddings:
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {"device": "cpu"}
    # encode_kwargs = {"clean_up_tokenization_spaces": True, "normalize_embeddings": False}
    encode_kwargs = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


# def _generate_vector_database(document_path: str, embedding_function: HuggingFaceEmbeddings, database_path: str, index_name: str):
#     split_documents = _ingest_pdf_documents(document_path)
#     return create_vector_database(split_documents, embedding_function, database_path, index_name)


_embedding_function =  _load_embedding_function()
_vector_database = restore_vector_database(_embedding_function, "")

app = fastapi.FastAPI()


@app.post("/query")
async def postgres_query(request: fastapi.Request, k: int = 1) -> PostgresRetrievalResponse:
    request_body = await request.body()
    query = request_body.decode()

    search_result = _vector_database.similarity_search(query=query, k=k)
    documents = [ SimilaritySearchDocument(content=document.page_content, source=document.metadata["source"], page=document.metadata["page"]) for document in search_result ]
    return PostgresRetrievalResponse(documents=documents)
