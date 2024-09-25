from pydantic import BaseModel
from typing import List


class SimilaritySearchDocument(BaseModel):
    content: str
    source: str
    page: int


class PostgresRetrievalResponse(BaseModel):
    documents: List[SimilaritySearchDocument]
