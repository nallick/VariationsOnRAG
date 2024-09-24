from langchain.indexes import index as langchain_index
from langchain.indexes import SQLRecordManager
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def _create_empty_faiss_database(embedding_function: HuggingFaceEmbeddings):
    import faiss  # macOS seg faults if this is imported too early
    from langchain_community.docstore.in_memory import InMemoryDocstore

    embedding_dimensions: int = len(embedding_function.embed_query("unused"))
    index = faiss.IndexFlatL2(embedding_dimensions)
    docstore = InMemoryDocstore()
    return FAISS(embedding_function=embedding_function, index=index, docstore=docstore, index_to_docstore_id={}, normalize_L2=False)


def _create_record_manager(database_path: str, collection_name: str):
    namespace = f"faiss/{collection_name}"
    return SQLRecordManager(namespace, db_url=f"sqlite:///{database_path}/record_manager_cache.sql")


def _faiss_index_name(collection_name: str):
    return f"{collection_name}-faiss_db"


def create_vector_database(split_documents, embedding_function: HuggingFaceEmbeddings, database_path: str, collection_name: str):
    database = _create_empty_faiss_database(embedding_function)
    record_manager = _create_record_manager(database_path, collection_name)
    record_manager.create_schema()
    index_result = langchain_index(split_documents, record_manager, database, cleanup=None, source_id_key="source")
    index_name = _faiss_index_name(collection_name)
    database.save_local(folder_path=database_path, index_name=index_name)
    return database, index_result


def restore_vector_database(embedding_function: HuggingFaceEmbeddings, database_path: str, collection_name: str):
    index_name = _faiss_index_name(collection_name)
    return FAISS.load_local(folder_path=database_path, embeddings=embedding_function, index_name=index_name, allow_dangerous_deserialization=True)


def update_vector_database(split_documents, database, database_path: str, collection_name: str, incremental: bool):
    record_manager = _create_record_manager(database_path, collection_name)
    cleanup = "incremental" if incremental else "full"
    return langchain_index(split_documents, record_manager, database, cleanup=cleanup, source_id_key="source")


def clear_vector_database(database, database_path: str, collection_name: str):
    record_manager = _create_record_manager(database_path, collection_name)
    return langchain_index([], record_manager, database, cleanup="full", source_id_key="source")
