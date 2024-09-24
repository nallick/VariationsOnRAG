from postgres_retrieval_service_types import *
from vector_database import *


_embedding_function = load_embedding_function()


def lambda_handler(event, context):
    k = event.get("queryStringParameters", {}).get("k", 1)
    query = event.get("body", "")

    document = SimilaritySearchDocument(content=str(type(_embedding_function)), source=query, page=k)
    return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": document.model_dump_json()}
