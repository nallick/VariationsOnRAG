import os

from langchain.indexes import index as index_vector_store
from langchain.indexes import SQLRecordManager
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector


def _connection_string() -> str:
    return PGVector.connection_string_from_db_params(                                                  
        driver = os.environ.get("PGVECTOR_DRIVER"),
        user = os.environ.get("PGVECTOR_USER"),                                      
        password = os.environ.get("PGVECTOR_PASSWORD"),                                  
        host = os.environ.get("PGVECTOR_HOST"),                                            
        port = os.environ.get("PGVECTOR_PORT"),                                          
        database = os.environ.get("PGVECTOR_DATABASE")
    )                                       


def load_embedding_function():
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


def _pgvector_database_connection(embedding_function: HuggingFaceEmbeddings, collection_name: str, connection_string: str):
    return PGVector(embeddings=embedding_function, collection_name=collection_name, connection=connection_string)


def _create_record_manager(collection_name: str, connection_string: str):
    namespace = f"postgres/{collection_name}"
    return SQLRecordManager(namespace, db_url=connection_string)


def create_vector_store(split_documents, embedding_function: HuggingFaceEmbeddings, unused_database_path: str, collection_name: str):
    connection_string = _connection_string()
    vector_store = _pgvector_database_connection(embedding_function, collection_name, connection_string)
    record_manager = _create_record_manager(collection_name, connection_string)
    record_manager.create_schema()
    index_result = index_vector_store(split_documents, record_manager, vector_store, cleanup=None, source_id_key="source")
    return vector_store, index_result


def restore_vector_store(embedding_function, unused_database_path, collection_name: str):
    connection_string = _connection_string()
    return _pgvector_database_connection(embedding_function, collection_name, connection_string)


def update_vector_store(split_documents, vector_store, unused_database_path: str, collection_name: str, incremental: bool):
    connection_string = _connection_string()
    record_manager = _create_record_manager(collection_name, connection_string)
    cleanup = "incremental" if incremental else "full"
    return index_vector_store(split_documents, record_manager, vector_store, cleanup=cleanup, source_id_key="source")


def clear_vector_store(vector_store, unused_database_path: str, collection_name: str):
    connection_string = _connection_string()
    record_manager = _create_record_manager(collection_name, connection_string)
    return index_vector_store([], record_manager, vector_store, cleanup="full", source_id_key="source")
