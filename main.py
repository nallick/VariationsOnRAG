from dotenv import load_dotenv
from enum import Enum
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

load_dotenv()


VECTOR_COLLECTION_NAME = "wwt_employee"

class VectorStore(Enum):
    FAISS = 1
    CHROMA = 2
    POSTGRES = 3
    AWS_LAMBDA = 4

VECTOR_STORE = VectorStore.FAISS

if VECTOR_STORE == VectorStore.FAISS:
    from vector_store_FAISS import *
elif VECTOR_STORE == VectorStore.CHROMA:
    from vector_store_Chroma import *
elif VECTOR_STORE == VectorStore.POSTGRES:
    from vector_store_PGVector import *
elif VECTOR_STORE == VectorStore.AWS_LAMBDA:
    from vector_store_AWS_Lambda import *
else:
    print("🛑 Unknown vector store specified")
    exit(0)


def _load_embedding_function() -> HuggingFaceEmbeddings:
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {"device": "cpu"}
    # encode_kwargs = {"clean_up_tokenization_spaces": True, "normalize_embeddings": False}
    encode_kwargs = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)


def _load_chat_model():
    import os
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

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


def _generate_vector_store(document_path: str, embedding_function: HuggingFaceEmbeddings, database_path: str):
    split_documents = _ingest_pdf_documents(document_path)
    return create_vector_store(split_documents, embedding_function, database_path, VECTOR_COLLECTION_NAME)


def _update_vector_store(database, document_path: str, database_path: str, incremental: bool):
    split_documents = _ingest_pdf_documents(document_path)
    return update_vector_store(split_documents, database, database_path, VECTOR_COLLECTION_NAME, incremental)


def _load_vector_store(embedding_function: HuggingFaceEmbeddings, database_path: str):
    try:
        return restore_vector_store(embedding_function, database_path, VECTOR_COLLECTION_NAME)
    except:
        return None


def _condense_response_sources(source_documents):
    unique_sources = { source.metadata["source"] for source in source_documents }
    source_dictionary = { source: set() for source in unique_sources }
    for source in source_documents:
        source_dictionary[source.metadata["source"]].add(source.metadata["page"])
    path_stem = lambda path_string : Path(path_string).stem
    return { path_stem(key): sorted(value) for (key, value) in source_dictionary.items() }


def _chatbot_input_loop(chat_model, vector_store, example_count, embedding_function: HuggingFaceEmbeddings, document_path: str, database_path: str):
    import time
    import langchain.hub as langchain_hub
    from langchain.chains import RetrievalQA

    def create_qa_chain():
        prompt = langchain_hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": example_count})
        chain_kwargs = {"prompt": prompt}
        return RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_kwargs)

    qa_chain = create_qa_chain() if vector_store else None

    print("\n🤖 Hello, how can I help you?")

    while True:
        try:
            query = input("\n🙂 ")
        except KeyboardInterrupt:
            break
        lowered_query = query.lower()
        if lowered_query in { "exit", "quit" }:
            break
        elif lowered_query in { "refresh", "update" }:
            update_result = _update_vector_store(vector_store, document_path, database_path, lowered_query == "refresh")
            print(f"📀 {update_result}")
        elif lowered_query == "clear":
            update_result = clear_vector_store(vector_store, database_path, VECTOR_COLLECTION_NAME)
            print(f"📀 {update_result}")
        elif lowered_query == "build":
            vector_store, build_result = _generate_vector_store(document_path, embedding_function, database_path)
            print(f"👷 {build_result}")
            qa_chain = create_qa_chain()
        else:
            if qa_chain:
                start_time = time.perf_counter()
                response = qa_chain.invoke({"query": query})
                end_time = time.perf_counter()
                sources = _condense_response_sources(response["source_documents"])
                print(f"🤖 {response['result']}")
                for source in sources:
                    print(f"    Source: {source}, page(s): {', '.join(str(page_number + 1) for page_number in sources[source])}")
                print(f"    ⏱️  {round(end_time - start_time, 2)} seconds")
            else:
                print("🛑 Vector store not found")


def main():
    chat_model = _load_chat_model()
    embedding_function =  _load_embedding_function()

    document_path = "Knowledge Base"
    database_path = "./Database"  # storage for local DBs
    vector_store = _load_vector_store(embedding_function, database_path)
    # search_result = vector_store.similarity_search("Does WWT celebrate Juneteenth?")
    # print("👹 ", search_result)

    example_count = 3
    _chatbot_input_loop(chat_model, vector_store, example_count, embedding_function, document_path, database_path)


if __name__ == "__main__":
    main()


# query = "Does WWT celebrate Juneteenth?"
# query = "Does WWT recognize Juneteenth?"
# query = "Do I get PTO for Juneteenth?"
# query = "How far ahead to I have to request time off?"
# query = "When do I need to submit my expense report?"
# query = "What can I use my company credit card for?"
# query = "Can I buy my boss a birthday present with my company credit card?"
