from transformers import AutoTokenizer, AutoModel


def save_pretrained_model(model_name: str, destination_path: str):
    # see: https://huggingface.co/docs/transformers/main/en/installation

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenizer.save_pretrained(destination_path)
    model.save_pretrained(destination_path)


if __name__ == "__main__":
    save_pretrained_model("sentence-transformers/all-MiniLM-l6-v2", "../all-MiniLM-l6-v2")
