# import os

# # from langchain.vectorstores.pgvector import PGVector (deprecated)
# from langchain_postgres import PGVector
# from langchain_huggingface import HuggingFaceEmbeddings


# PGVECTOR_CONNECTION_STRING = PGVector.connection_string_from_db_params(                                                  
#     driver = os.environ.get("PGVECTOR_DRIVER"),
#     user = os.environ.get("PGVECTOR_USER"),                                      
#     password = os.environ.get("PGVECTOR_PASSWORD"),                                  
#     host = os.environ.get("PGVECTOR_HOST"),                                            
#     port = os.environ.get("PGVECTOR_PORT"),                                          
#     database = os.environ.get("PGVECTOR_DATABASE")
# )                                       


# def create_vector_database(split_documents, embedding_function: HuggingFaceEmbeddings, unused_database_path: str):
#     # return PGVector.from_documents(split_documents, embedding=embedding_function, connection_string=PGVECTOR_CONNECTION_STRING)
#     return PGVector.from_documents(split_documents, embedding=embedding_function, connection=PGVECTOR_CONNECTION_STRING)


# def restore_vector_database(embedding_function: HuggingFaceEmbeddings, unused_database_path: str):
#     # return PGVector(connection_string=PGVECTOR_CONNECTION_STRING, embedding_function=embedding_function)
#     return PGVector(connection=PGVECTOR_CONNECTION_STRING, embeddings=embedding_function)


import requests

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import TypeAdapter
from typing import List

from postgres_retrieval_service_types import *


class PostgresServiceRetriever(BaseRetriever):
    k: int

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_url = f"http://127.0.0.1:8000/query?k={self.k}"
        api_response = requests.post(query_url, data=str.encode(query), headers={"Content-Type": "application/octet-stream"})
        query_result = TypeAdapter(PostgresRetrievalResponse).validate_json(api_response.content)
        result = [ Document(page_content=document.content, metadata={"source": document.source, "page": document.page}) for document in query_result.documents ]
        return result

    # async def _aget_relevant_documents(self, query: str) -> List[Document]:
    #     """(Optional) async native implementation."""
    #     return self.docs[:self.k]


def restore_vector_database(unused_embedding_function, unused_database_path):
    return None


def create_retriever(example_count):
    return PostgresServiceRetriever(k=example_count)