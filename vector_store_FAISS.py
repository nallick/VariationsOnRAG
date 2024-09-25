from langchain.indexes import index as index_vector_store
from langchain.indexes import SQLRecordManager
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def _create_empty_faiss_vector_store(embedding_function: HuggingFaceEmbeddings):
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


def create_vector_store(split_documents, embedding_function: HuggingFaceEmbeddings, database_path: str, collection_name: str):
    vector_store = _create_empty_faiss_vector_store(embedding_function)
    record_manager = _create_record_manager(database_path, collection_name)
    record_manager.create_schema()
    index_result = index_vector_store(split_documents, record_manager, vector_store, cleanup=None, source_id_key="source")
    index_name = _faiss_index_name(collection_name)
    vector_store.save_local(folder_path=database_path, index_name=index_name)
    return vector_store, index_result


def restore_vector_store(embedding_function: HuggingFaceEmbeddings, database_path: str, collection_name: str):
    index_name = _faiss_index_name(collection_name)
    return FAISS.load_local(folder_path=database_path, embeddings=embedding_function, index_name=index_name, allow_dangerous_deserialization=True)


def update_vector_store(split_documents, vector_store, database_path: str, collection_name: str, incremental: bool):
    record_manager = _create_record_manager(database_path, collection_name)
    cleanup = "incremental" if incremental else "full"
    return index_vector_store(split_documents, record_manager, vector_store, cleanup=cleanup, source_id_key="source")


def clear_vector_store(vector_store, database_path: str, collection_name: str):
    record_manager = _create_record_manager(database_path, collection_name)
    return index_vector_store([], record_manager, vector_store, cleanup="full", source_id_key="source")
