from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path


def _load_embedding_function() -> HuggingFaceEmbeddings:
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {"device": "cpu"}
    # encode_kwargs = {"clean_up_tokenization_spaces": True, "normalize_embeddings": False}
    encode_kwargs = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


def _load_chat_model():
    import os
    from dotenv import load_dotenv
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    load_dotenv()
    base_url = os.environ["NVIDIA_API_URL"]
    api_key = os.environ["NVIDIA_API_KEY"]
    llm_model = os.environ["LLM_MODEL"]
    return ChatNVIDIA(base_url=base_url, api_key=api_key, model=llm_model, max_tokens=175)


def _ingest_pdf_documents(document_path: str):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFDirectoryLoader

    pdf_loader = PyPDFDirectoryLoader(document_path)
    documents = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)


def _generate_vector_database(document_path: str, embedding_function: HuggingFaceEmbeddings, database_path: str, index_name: str):
    split_documents = _ingest_pdf_documents(document_path)
    database = FAISS.from_documents(split_documents, embedding_function)
    database.save_local(folder_path=database_path, index_name=index_name)


def _load_vector_database(embedding_function: HuggingFaceEmbeddings, database_path: str, index_name: str):
    return FAISS.load_local(folder_path=database_path, embeddings=embedding_function, index_name=index_name, allow_dangerous_deserialization=True)


def _condense_response_sources(source_documents):
    unique_sources = { source.metadata['source'] for source in source_documents }
    source_dictionary = { source: set() for source in unique_sources }
    for source in source_documents:
        source_dictionary[source.metadata['source']].add(source.metadata['page'])
    path_stem = lambda path_string : Path(path_string).stem
    return { path_stem(key): sorted(value) for (key, value) in source_dictionary.items() }


def _chatbot_input_loop(chat_model, vector_database, example_count):
    import langchain.hub as langchain_hub
    from langchain.chains import RetrievalQA

    prompt = langchain_hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
    retriever = vector_database.as_retriever(search_type="similarity", search_kwargs={"k": example_count})
    chain_kwargs = {"prompt": prompt}
    qa_chain = RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_kwargs)

    print("\n🤖 Hello, how can I help you?")

    while True:
        try:
            query = input("\n🙂 ")
        except KeyboardInterrupt:
            break
        response = qa_chain.invoke({"query": query})
        sources = _condense_response_sources(response["source_documents"])
        print(f"🤖 {response['result']}")
        for source in sources:
            print(f"    Source: {source}, page(s): {', '.join(str(page_number) for page_number in sources[source])}")


def main():
    chat_model = _load_chat_model()
    embedding_function =  _load_embedding_function()

    database_path = "./Database"
    database_index = "WWT_Index"
    # _generate_database("Knowledge Base", embedding_function, database_path, database_index)
    vector_database = _load_vector_database(embedding_function, database_path, database_index)
    # search_result = vector_database.similarity_search(query)

    example_count = 3
    _chatbot_input_loop(chat_model, vector_database, example_count)


if __name__ == "__main__":
    main()


# query = "Does WWT celebrate Juneteenth?"
# query = "Does WWT recognize Juneteenth?"
# query = "Do I get PTO for Juneteenth?"
# query = "How far ahead to I have to request time off?"
# query = "When do I need to submit my expense report?"
# query = "What can I use my company credit card for?"
# query = "Can I buy my boss a birthday present with my company credit card?"
