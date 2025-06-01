import logging
import uuid
import urllib.parse
import io
import pymupdf4llm
import pymupdf  # Re-import pymupdf as it's needed for pymupdf.open()
import fitz      # Import fitz for PyMuPDF core functionalities and exceptions
from markitdown import MarkItDown
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter

from celery import Task
from celery_app import celery_app
from app.core.clients import get_supabase_client, get_openai_client
from app.core.config import (
    PARENT_CHUNK_SIZE, 
    PARENT_CHUNK_OVERLAP, 
    CHILD_CHUNK_SIZE, 
    CHILD_CHUNK_OVERLAP,
    EMBEDDING_MODEL
)

class RAGProcessorTask(Task):
    """Custom Celery Task that keeps clients initialized across task invocations"""
    _supabase_client = None
    _openai_client = None
    
    @property
    def supabase_client(self):
        if self._supabase_client is None:
            self._supabase_client = get_supabase_client()
        return self._supabase_client
    
    @property
    def openai_client(self):
        if self._openai_client is None:
            self._openai_client = get_openai_client()
        return self._openai_client

@celery_app.task(bind=True, base=RAGProcessorTask, name="process_material")
def process_material_task(self, material_id, storage_path):
    """
    Process a material document for RAG, including:
    1. Download from Supabase Storage
    2. Convert to text/markdown
    3. Chunk the document
    4. Generate embeddings
    5. Store in the database
    """
    logging.info(f"[{material_id}] Starting material processing in Celery task...")
    
    # Ensure clients are available
    if not self.supabase_client or not self.openai_client:
        error_msg = "Task cannot run: Clients not initialized."
        logging.error(f"[{material_id}] {error_msg}")
        
        # Attempt to update status to FAILED even if clients weren't fully ready
        if self.supabase_client and material_id:
            try:
                self.supabase_client.table('materials').update({
                    "status": 'FAILED',
                    "errorMessage": "Server configuration error (clients unavailable)"
                }).eq('id', material_id).execute()
            except Exception as db_err:
                logging.exception(f"[{material_id}] CRITICAL: Failed to update status to FAILED (clients unavailable).")
        return # Exit the task
    
    # Instantiate MarkItDown converter
    converter = MarkItDown()
    logging.info(f"[{material_id}] MarkItDown converter initialized")
    
    # Initialize success/error state
    success = False
    error_message = ""
    extracted_text = None
    
    try:
        logging.info(f"[{material_id}] Downloading document from storage: {storage_path}")
        
        # --- First update status to PROCESSING ---
        self.supabase_client.table('materials').update({
            "status": 'PROCESSING',
            "errorMessage": None
        }).eq('id', material_id).execute()
        
        # --- Download Document from Storage ---
        bucket_name = "materials"
        file_path = storage_path.replace("storage/v1/object/public/materials/", "")
        # URL decode file_path to handle spaces and special characters
        file_path = urllib.parse.unquote(file_path)
        
        logging.info(f"[{material_id}] Retrieving file {file_path} from bucket {bucket_name}")
        
        # Get file data from Supabase Storage
        response = self.supabase_client.storage.from_(bucket_name).download(file_path)
        
        if not response:
            raise Exception(f"Failed to download file from Supabase Storage: {file_path}")
        
        # --- Process Document ---
        file_data = io.BytesIO(response)
        
        # Convert document to markdown using pymupdf4llm
        logging.info(f"[{material_id}] Converting document to markdown...")
        
        try:
            # First try pymupdf4llm which handles PDF, DOCX, etc. and produces nice Markdown
            extracted_text = pymupdf4llm.to_markdown(file_data)
            logging.info(f"[{material_id}] Successfully converted document using pymupdf4llm")
        except Exception as convert_err:
            logging.exception(f"[{material_id}] Failed to convert with pymupdf4llm, falling back to PyMuPDF: {convert_err}")
            
            try:
                # Fallback to basic PyMuPDF for PDFs only
                file_data.seek(0)  # Reset position in file for re-reading
                pdf_document = fitz.open(stream=file_data, filetype="pdf")
                text_parts = []
                
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    text_parts.append(page.get_text())
                
                extracted_text = "\n".join(text_parts)
                pdf_document.close()
                logging.info(f"[{material_id}] Successfully converted document using PyMuPDF fallback")
                
                # Convert plain text to Markdown using MarkItDown
                logging.info(f"[{material_id}] Converting plain text to markdown with MarkItDown...")
                extracted_text = converter.convert(extracted_text)
            except Exception as fallback_err:
                raise Exception(f"Failed to extract text with both pymupdf4llm and PyMuPDF: {fallback_err}")
        
        # --- Process and Store Chunks ---
        if extracted_text:
            logging.info(f"[{material_id}] Extracted text length: {len(extracted_text)} characters")
            
            # Generate parent chunks (larger chunks for context)
            parent_splitter = MarkdownTextSplitter(
                chunk_size=PARENT_CHUNK_SIZE, 
                chunk_overlap=PARENT_CHUNK_OVERLAP
            )
            parent_chunks = parent_splitter.split_text(extracted_text)
            
            logging.info(f"[{material_id}] Created {len(parent_chunks)} parent chunks")
            
            # Generate embeddings for all parent chunks
            all_vectors = []
            
            # Process each parent chunk to create embeddings and store
            for i, parent_chunk in enumerate(parent_chunks):
                logging.info(f"[{material_id}] Processing parent chunk {i+1}/{len(parent_chunks)}")
                
                # Further split into smaller chunks for embeddings (child chunks)
                child_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHILD_CHUNK_SIZE,
                    chunk_overlap=CHILD_CHUNK_OVERLAP
                )
                child_chunks = child_splitter.split_text(parent_chunk)
                
                for j, child_text in enumerate(child_chunks):
                    # Generate embedding
                    embedding_response = self.openai_client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=child_text
                    )
                    
                    # Get the embedding vector
                    embedding = embedding_response.data[0].embedding
                    
                    # Prepare vector record for Supabase
                    vector_record = {
                        "id": str(uuid.uuid4()),
                        "material_id": material_id,
                        "content": child_text,
                        "embedding": embedding,
                        "parent_chunk_index": i,
                        "child_chunk_index": j,
                        "parent_chunk": parent_chunk if j == 0 else None  # Only store parent content on first child
                    }
                    
                    all_vectors.append(vector_record)
            
            # Store all vector records in Supabase
            logging.info(f"[{material_id}] Storing {len(all_vectors)} vector records in Supabase...")
            
            # Batch insert vectors (in groups of 50 to avoid request size limits)
            batch_size = 50
            for i in range(0, len(all_vectors), batch_size):
                batch = all_vectors[i:i+batch_size]
                self.supabase_client.table('material_embeddings').insert(batch).execute()
            
            # --- Update Material Status ---
            logging.info(f"[{material_id}] Processing complete, updating material status to COMPLETE")
            self.supabase_client.table('materials').update({
                "status": 'COMPLETE',
                "errorMessage": None,
                "totalChunks": len(all_vectors)
            }).eq('id', material_id).execute()
            
            success = True
            logging.info(f"[{material_id}] Successfully processed material")
        else:
            error_message = "Failed to extract text from document"
            logging.error(f"[{material_id}] {error_message}")
            
    except Exception as e:
        error_message = f"Error processing material: {str(e)}"
        logging.exception(f"[{material_id}] {error_message}")
        
        # Update material status to FAILED
        try:
            if self.supabase_client and material_id:
                self.supabase_client.table('materials').update({
                    "status": 'FAILED',
                    "errorMessage": error_message[:900]  # Truncate to avoid DB field size issues
                }).eq('id', material_id).execute()
        except Exception as update_err:
            logging.exception(f"[{material_id}] Failed to update material status: {update_err}")
    
    return {
        "material_id": material_id,
        "success": success,
        "error": error_message
    }
