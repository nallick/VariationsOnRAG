from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_database(split_documents, embedding_function: HuggingFaceEmbeddings, database_path: str, index_name: str):
    database = FAISS.from_documents(split_documents, embedding_function)
    database.save_local(folder_path=database_path, index_name=index_name)
    return database


def restore_vector_database(embedding_function: HuggingFaceEmbeddings, database_path: str, index_name: str):
    return FAISS.load_local(folder_path=database_path, embeddings=embedding_function, index_name=index_name, allow_dangerous_deserialization=True)
