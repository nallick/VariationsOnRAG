import boto3, json

from langchain_core.embeddings import Embeddings
from typing import List


class BedrockEmbeddings(Embeddings):

    def __init__(self, model: str="amazon.titan-embed-text-v2:0", region: str="us-east-1", vector_dimensions: int=1024):
        self.model = model
        self.vector_dimensions = vector_dimensions  # titan-embed-text-v2 supports 256 - 1024
        self.bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=region)


    def embed_query(self, query: str) -> List[float]:
        model_input = { "inputText": query, "dimensions": self.vector_dimensions, "normalize": True }
        request_body = json.dumps(model_input)
        response = self.bedrock_client.invoke_model(body=request_body, modelId=self.model, accept="application/json", contentType="application/json")
        response_body = json.loads(response.get("body").read())
        return response_body.get("embedding")


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]
