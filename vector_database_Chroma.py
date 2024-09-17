from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_database(split_documents, embedding_function: HuggingFaceEmbeddings, database_path: str, unused_index_name: str):
    database = Chroma.from_documents(split_documents, embedding=embedding_function, persist_directory=database_path)
    return database


def restore_vector_database(embedding_function: HuggingFaceEmbeddings, database_path: str, index_name: str):
    return Chroma(persist_directory=database_path, embedding_function=embedding_function)
