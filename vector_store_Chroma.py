from langchain.indexes import index as index_vector_store
from langchain.indexes import SQLRecordManager
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def _create_empty_chroma_store(embedding_function: HuggingFaceEmbeddings, collection_name: str, persist_directory: str):
    return Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=embedding_function)


def _create_record_manager(database_path: str, collection_name: str):
    namespace = f"chroma/{collection_name}"
    return SQLRecordManager(namespace, db_url=f"sqlite:///{database_path}/record_manager_cache.sql")


def create_vector_store(split_documents, embedding_function: HuggingFaceEmbeddings, database_path: str, collection_name: str):
    vector_store = _create_empty_chroma_store(embedding_function, collection_name, database_path)
    record_manager = _create_record_manager(database_path, collection_name)
    record_manager.create_schema()
    index_result = index_vector_store(split_documents, record_manager, vector_store, cleanup=None, source_id_key="source")
    return vector_store, index_result


def restore_vector_store(embedding_function: HuggingFaceEmbeddings, database_path: str, collection_name: str):
    return Chroma(persist_directory=database_path, collection_name=collection_name, embedding_function=embedding_function)


def update_vector_store(split_documents, vector_store, database_path: str, collection_name: str, incremental: bool):
    record_manager = _create_record_manager(database_path, collection_name)
    cleanup = "incremental" if incremental else "full"
    return index_vector_store(split_documents, record_manager, vector_store, cleanup=cleanup, source_id_key="source")


def clear_vector_store(vector_store, database_path: str, collection_name: str):
    record_manager = _create_record_manager(database_path, collection_name)
    return index_vector_store([], record_manager, vector_store, cleanup="full", source_id_key="source")
