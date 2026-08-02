import os 
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv

load_dotenv()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

def load_and_store_documents(docs_folder: str) -> list:
    # Load the documents
    #loader = TextLoader(file_path)
    #documents = loader.load()

    # Split the document into chunks
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len)
    #chunks = text_splitter.split_documents(documents)
    docs_path = Path(docs_folder)
    for txt_file in docs_path.glob("*.txt"):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        print(f"Processed {len(chunks)} chunks from {txt_file.name}")
    return all_chunks

def ingest_to_chromadb(chunks: list) -> None:
    # Clear existing collection items to ensure clean ingestion with updated metadata
    existing = collection.get()
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])
        
    # Store the chunks in ChromaDB with source metadata
    for i, chunk in enumerate(chunks):
        source_path = chunk.metadata.get("source", "unknown")
        source_filename = Path(source_path).name
        collection.add(
            documents=[chunk.page_content],
            metadatas={"source": source_filename},
            ids=[f"chunk_{i}"]
        )
    print(f"Ingested {len(chunks)} chunks into ChromaDB with source metadata.")

if __name__ == "__main__":
    print("starting document ingestion...")
    chunks = load_and_store_documents("docs")
    if not chunks:
        print("No chunks were created. Please check the documents in the 'docs' folder.")
    else:
        ingest_to_chromadb(chunks)
        print("Document ingestion completed successfully, total chunks ingested: ", len(chunks))
        print("Your knowledge base is ready to use!")
        