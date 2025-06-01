from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from app.core.config import (
    PARENT_CHUNK_SIZE, 
    PARENT_CHUNK_OVERLAP, 
    CHILD_CHUNK_SIZE, 
    CHILD_CHUNK_OVERLAP
)

def create_parent_chunks(text):
    """
    Split markdown text into parent chunks (larger context pieces)
    
    Args:
        text (str): The markdown text to split
        
    Returns:
        list: List of parent chunk strings
    """
    parent_splitter = MarkdownTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, 
        chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    return parent_splitter.split_text(text)

def create_child_chunks(parent_chunk):
    """
    Split a parent chunk into smaller child chunks for embedding
    
    Args:
        parent_chunk (str): The parent chunk text to split
        
    Returns:
        list: List of child chunk strings
    """
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    return child_splitter.split_text(parent_chunk)
