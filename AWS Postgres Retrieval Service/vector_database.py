import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector


PGVECTOR_CONNECTION_STRING = PGVector.connection_string_from_db_params(                                                  
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


# def create_vector_database(split_documents, embedding_function: HuggingFaceEmbeddings, unused_database_path: str):
#     # return PGVector.from_documents(split_documents, embedding=embedding_function, connection_string=PGVECTOR_CONNECTION_STRING)
#     return PGVector.from_documents(split_documents, embedding=embedding_function, connection=PGVECTOR_CONNECTION_STRING)


# def restore_vector_database(embedding_function: HuggingFaceEmbeddings, unused_database_path: str):
#     # return PGVector(connection_string=PGVECTOR_CONNECTION_STRING, embedding_function=embedding_function)
#     return PGVector(connection=PGVECTOR_CONNECTION_STRING, embeddings=embedding_function)
