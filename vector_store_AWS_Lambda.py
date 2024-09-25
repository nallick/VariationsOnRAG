import requests

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import TypeAdapter
from typing import List

from postgres_retrieval_service_types import *


class LambdaEmployeeInfoServiceRetriever(BaseRetriever):
    k: int

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_url = f"https://lfl6sjjpkfkq2agkrm3if427pu0fohpn.lambda-url.us-east-1.on.aws?k={self.k}"
        api_response = requests.post(query_url, data=str.encode(query), headers={"Content-Type": "text/plain"})
        query_result = TypeAdapter(PostgresRetrievalResponse).validate_json(api_response.content)
        result = [ Document(page_content=document.content, metadata={"source": document.source, "page": document.page}) for document in query_result.documents ]
        return result

    # async def _aget_relevant_documents(self, query: str) -> List[Document]:
    #     """(Optional) async native implementation."""
    #     return self.docs[:self.k]


class LambdaEmployeeInfoProxy():
    def as_retriever(self, search_type, search_kwargs):
        return LambdaEmployeeInfoServiceRetriever(k=search_kwargs["k"])


def restore_vector_store(embedding_function, unused_database_path, collection_name: str):
    return LambdaEmployeeInfoProxy()
