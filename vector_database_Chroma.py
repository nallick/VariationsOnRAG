from langchain.indexes import index as langchain_index
from langchain.indexes import SQLRecordManager
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def _create_empty_chroma_database(embedding_function: HuggingFaceEmbeddings, collection_name: str, persist_directory: str):
    return Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=embedding_function)


def _create_record_manager(database_path: str, collection_name: str):
    namespace = f"chroma/{collection_name}"
    return SQLRecordManager(namespace, db_url=f"sqlite:///{database_path}/record_manager_cache.sql")


def create_vector_database(split_documents, embedding_function: HuggingFaceEmbeddings, database_path: str, collection_name: str):
    database = _create_empty_chroma_database(embedding_function, collection_name, database_path)
    record_manager = _create_record_manager(database_path, collection_name)
    record_manager.create_schema()
    index_result = langchain_index(split_documents, record_manager, database, cleanup=None, source_id_key="source")
    return database, index_result


def restore_vector_database(embedding_function: HuggingFaceEmbeddings, database_path: str, collection_name: str):
    return Chroma(persist_directory=database_path, collection_name=collection_name, embedding_function=embedding_function)


def update_vector_database(split_documents, database, database_path: str, collection_name: str, incremental: bool):
    record_manager = _create_record_manager(database_path, collection_name)
    cleanup = "incremental" if incremental else "full"
    return langchain_index(split_documents, record_manager, database, cleanup=cleanup, source_id_key="source")


def clear_vector_database(database, database_path: str, collection_name: str):
    record_manager = _create_record_manager(database_path, collection_name)
    return langchain_index([], record_manager, database, cleanup="full", source_id_key="source")
