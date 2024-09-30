import torch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
from typing import List


class OfflineEmbeddings(Embeddings):

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)


    def _encode(self, text) -> torch.Tensor:
        # see: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/README.md#usage-huggingface-transformers

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        if isinstance(text, Document):
            text = text.page_content
        if not isinstance(text, str):
            print("🤡 ", type(text), text)
            raise ValueError(f"text to encode must be str or Document, not {type(text)}")
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        result = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return result[0]


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._encode(text).tolist() for text in texts]


    def embed_query(self, query: str) -> List[float]:
        return self._encode(query).tolist()


    @staticmethod
    def save_pretrained_model(model_name: str, destination_path: str):
        # see: https://huggingface.co/docs/transformers/main/en/installation

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        tokenizer.save_pretrained(destination_path)
        model.save_pretrained(destination_path)