from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

# Project root directory
PROJECT_ROOT = Path(__file__).parent
CHROMA_DIR = PROJECT_ROOT / ".chroma"
COLLECTION_NAME = "rag-chroma"


def build_vector_db(
    pdf_path: Path,
    persist_dir: Path,
    collection_name: str,
    section_delimiter: str = "skirtukas"
) -> Chroma:
    """
    Build a vector database from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
        persist_dir: Directory to persist the vector database.
        collection_name: Name of the collection.
        section_delimiter: Text to split sections on.
        
    Returns:
        Chroma vector store instance.
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        ValueError: If the PDF has no content.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    
    if not pages:
        raise ValueError(f"No pages found in PDF: {pdf_path}")
    
    full_text = "\n".join(page.page_content for page in pages)
    full_text = " ".join(full_text.split())
    
    sections = [section.strip() for section in full_text.split(section_delimiter)]
    # Filter out empty sections
    sections = [s for s in sections if s]

    documents = [
        Document(
            page_content=section,
            metadata={"section_index": index},
            # Could add sources if needed
            # metadata={"source": str(pdf_path.name), "section_index": index},
        )
        for index, section in enumerate(sections)
    ]
    
    if not documents:
        raise ValueError("No documents created from PDF content")
    
    return Chroma.from_documents(
        documents=documents,
        collection_name=collection_name,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory=str(persist_dir),
    )


def get_retriever(
    persist_dir: Optional[Path] = None,
    collection_name: str = COLLECTION_NAME,
    k: int = 4
) -> VectorStoreRetriever:
    """
    Get a retriever from the existing vector database.
    
    Args:
        persist_dir: Directory where the vector database is persisted.
        collection_name: Name of the collection.
        k: Number of documents to retrieve.
        
    Returns:
        A retriever instance.
    """
    persist_dir = persist_dir or CHROMA_DIR
    
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
    ).as_retriever(search_kwargs={"k": k})


# Default retriever instance
retriever = get_retriever()


def main() -> None:
    """Ingest the PDF and build the vector database."""
    pdf_path = PROJECT_ROOT / "data" / "manobustas-paslaugos.pdf"
    
    vector_db = build_vector_db(pdf_path, CHROMA_DIR, COLLECTION_NAME)
    count = vector_db._collection.count()
    print(f"Successfully created vector database with {count} documents")


if __name__ == "__main__":
    main()