from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


FAISS_INDEX_NAME = "faiss_db"


def create_vector_database(split_documents, embedding_function: HuggingFaceEmbeddings, database_path: str):
    database = FAISS.from_documents(split_documents, embedding_function)
    database.save_local(folder_path=database_path, index_name=FAISS_INDEX_NAME)
    return database


def restore_vector_database(embedding_function: HuggingFaceEmbeddings, database_path: str):
    return FAISS.load_local(folder_path=database_path, embeddings=embedding_function, index_name=FAISS_INDEX_NAME, allow_dangerous_deserialization=True)
