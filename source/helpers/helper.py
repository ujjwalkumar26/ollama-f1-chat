#!/usr/bin/env python3
"""
Helper script to ingest documents into ChromaDB vector store.
Supports text files (.txt, .md) and PDF files (.pdf).

Usage (from project root):
    python source/helpers/helper.py <file_or_directory_path>
    python source/helpers/helper.py documents/
    python source/helpers/helper.py documents/my_file.pdf
"""

import sys
import logging
from pathlib import Path
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# Configuration - ChromaDB at project root (two levels up from this file)
CHROMA_DB_PATH = str(Path(__file__).parent.parent.parent / "chroma_db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf'}

# Initialize embeddings using standard all-MiniLM-L6-v2
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


def load_single_file(file_path: Path) -> List[Document]:
    """Load a single file based on its extension"""
    extension = file_path.suffix.lower()
    
    if extension == '.pdf':
        logger.info(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(str(file_path))
    elif extension in {'.txt', '.md'}:
        logger.info(f"Loading text file: {file_path}")
        loader = TextLoader(str(file_path), encoding='utf-8')
    else:
        logger.warning(f"Unsupported file type: {extension}")
        return []
    
    try:
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document(s) from {file_path.name}")
        return documents
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return []


def load_directory(dir_path: Path) -> List[Document]:
    """Load all supported files from a directory"""
    logger.info(f"Scanning directory: {dir_path}")
    
    all_documents = []
    
    # Process each supported file type
    for extension in SUPPORTED_EXTENSIONS:
        pattern = f"**/*{extension}"
        files = list(dir_path.glob(pattern))
        
        if files:
            logger.info(f"Found {len(files)} {extension} file(s)")
            for file_path in files:
                docs = load_single_file(file_path)
                all_documents.extend(docs)
    
    return all_documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    logger.info(f"Splitting {len(documents)} document(s) into chunks...")
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    return chunks


def ingest_to_chroma(chunks: List[Document]):
    """Ingest document chunks into ChromaDB"""
    if not chunks:
        logger.warning("No chunks to ingest!")
        return
    
    logger.info(f"Ingesting {len(chunks)} chunks into ChromaDB at {CHROMA_DB_PATH}")
    
    try:
        # Create or load vector store
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name="personal_knowledge"
        )
        
        # Add documents
        vectorstore.add_documents(chunks)
        
        logger.info("✅ Successfully ingested documents into ChromaDB")
        logger.info(f"Total documents in collection: {vectorstore._collection.count()}")
        
    except Exception as e:
        logger.error(f"Error ingesting to ChromaDB: {str(e)}", exc_info=True)
        sys.exit(1)


def show_collection_stats():
    """Display statistics about the current collection"""
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name="personal_knowledge"
        )
        
        count = vectorstore._collection.count()
        logger.info(f"Current collection size: {count} documents")
        
    except Exception as e:
        logger.error(f"Error reading collection: {str(e)}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python helper.py <file_or_directory_path>")
        print("\nExamples:")
        print("  python helper.py documents/")
        print("  python helper.py documents/my_file.pdf")
        print("  python helper.py notes.txt")
        print("\nSupported file types: .txt, .md, .pdf")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        logger.error(f"Path does not exist: {input_path}")
        sys.exit(1)
    
    # Show current stats
    logger.info("=" * 60)
    logger.info("ChromaDB Knowledge Ingestion Tool")
    logger.info("=" * 60)
    show_collection_stats()
    logger.info("=" * 60)
    
    # Load documents
    documents = []
    
    if input_path.is_file():
        documents = load_single_file(input_path)
    elif input_path.is_dir():
        documents = load_directory(input_path)
    else:
        logger.error(f"Invalid path: {input_path}")
        sys.exit(1)
    
    if not documents:
        logger.warning("No documents were loaded!")
        sys.exit(1)
    
    # Split into chunks
    chunks = split_documents(documents)
    
    # Ingest into ChromaDB
    ingest_to_chroma(chunks)
    
    # Show updated stats
    logger.info("=" * 60)
    show_collection_stats()
    logger.info("=" * 60)


if __name__ == "__main__":
    main()