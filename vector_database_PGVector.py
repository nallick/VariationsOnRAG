import os
import requests

from langchain.indexes import index as langchain_index
from langchain.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
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


class PGVectorProxy():
    def as_retriever(self, search_type, search_kwargs):
        return PostgresServiceRetriever(k=search_kwargs["k"])


def _connection_string() -> str:
    return PGVector.connection_string_from_db_params(                                                  
        driver = os.environ.get("PGVECTOR_DRIVER"),
        user = os.environ.get("PGVECTOR_USER"),                                      
        password = os.environ.get("PGVECTOR_PASSWORD"),                                  
        host = os.environ.get("PGVECTOR_HOST"),                                            
        port = os.environ.get("PGVECTOR_PORT"),                                          
        database = os.environ.get("PGVECTOR_DATABASE")
    )                                       


def _pgvector_database_connection(embedding_function: HuggingFaceEmbeddings, collection_name: str, connection_string: str):
    return PGVector(embeddings=embedding_function, collection_name=collection_name, connection=connection_string)


def _create_record_manager(collection_name: str, connection_string: str):
    namespace = f"postgres/{collection_name}"
    return SQLRecordManager(namespace, db_url=connection_string)


def create_vector_database(split_documents, embedding_function: HuggingFaceEmbeddings, unused_database_path: str, collection_name: str):
    # return PGVector.from_documents(split_documents, embedding=embedding_function, connection=PGVECTOR_CONNECTION_STRING)
    connection_string = _connection_string()
    database = _pgvector_database_connection(embedding_function, collection_name, connection_string)
    record_manager = _create_record_manager(collection_name, connection_string)
    record_manager.create_schema()
    index_result = langchain_index(split_documents, record_manager, database, cleanup=None, source_id_key="source")
    return database, index_result


def restore_vector_database(embedding_function, unused_database_path, collection_name: str):
    connection_string = _connection_string()
    return _pgvector_database_connection(embedding_function, collection_name, connection_string)
    # return PGVectorProxy()


def update_vector_database(split_documents, database, unused_database_path: str, collection_name: str, incremental: bool):
    connection_string = _connection_string()
    record_manager = _create_record_manager(collection_name, connection_string)
    cleanup = "incremental" if incremental else "full"
    return langchain_index(split_documents, record_manager, database, cleanup=cleanup, source_id_key="source")


def clear_vector_database(database, unused_database_path: str, collection_name: str):
    connection_string = _connection_string()
    record_manager = _create_record_manager(collection_name, connection_string)
    return langchain_index([], record_manager, database, cleanup="full", source_id_key="source")
