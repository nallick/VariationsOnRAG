from dotenv import load_dotenv
from postgres_retrieval_service_types import *
from vector_store import *


VECTOR_COLLECTION_NAME = "wwt_employee"

load_dotenv()
_embedding_function = load_embedding_function()
_vector_store = restore_vector_store(_embedding_function, "", VECTOR_COLLECTION_NAME)


def lambda_handler(event, context):
    k = event.get("queryStringParameters", {}).get("k", 1)
    query = event.get("body", "")

    search_result = _vector_store.similarity_search(query=query, k=k)
    documents = [ SimilaritySearchDocument(content=document.page_content, source=document.metadata["source"], page=document.metadata["page"]) for document in search_result ]
    response = PostgresRetrievalResponse(documents=documents)
    return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": response.model_dump_json()}
