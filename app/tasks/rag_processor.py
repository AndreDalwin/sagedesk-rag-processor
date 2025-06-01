import logging
import uuid
import urllib.parse
import io
import os
import tempfile
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
    EMBEDDING_MODEL,
    CELERY_BROKER_URL
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
    logging.critical(f"[{material_id}] WORKER START: Beginning task execution on worker node")
    logging.critical(f"[{material_id}] Broker URL: {CELERY_BROKER_URL}")
    logging.critical(f"[{material_id}] Storage path to process: {storage_path}")
    logging.info(f"[{material_id}] Starting material processing in Celery task...")
    
    # Ensure clients are available
    logging.critical(f"[{material_id}] Initializing clients...")
    try:
        supabase = self.supabase_client
        openai = self.openai_client
        logging.critical(f"[{material_id}] Clients initialized successfully: Supabase={bool(supabase)}, OpenAI={bool(openai)}")
    except Exception as client_err:
        logging.critical(f"[{material_id}] ERROR initializing clients: {str(client_err)}")
        
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
        logging.critical(f"[{material_id}] WORKER PHASE 1: Downloading document from storage: {storage_path}")
        
        # --- First update status to PROCESSING ---
        logging.critical(f"[{material_id}] Updating status to PROCESSING in Supabase...")
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
        temp_file_path = None # Initialize for the finally block
        try:
            # Create a temporary file to store the downloaded content
            # Using a suffix like .pdf can help libraries identify the file type, though pymupdf4llm is often robust.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(response) # response contains bytes from Supabase download
                temp_file_path = temp_file.name
            logging.info(f"[{material_id}] Document downloaded to temporary file: {temp_file_path}")

            # Convert document to markdown using pymupdf4llm
            logging.critical(f"[{material_id}] WORKER PHASE 2: Converting document to markdown...")
            
            try:
                # First try pymupdf4llm which handles PDF, DOCX, etc. and produces nice Markdown
                # Pass the path to the temporary file
                extracted_text = pymupdf4llm.to_markdown(temp_file_path)
                logging.info(f"[{material_id}] Successfully converted document using pymupdf4llm")
            except Exception as convert_err:
                logging.exception(f"[{material_id}] Failed to convert with pymupdf4llm, falling back to PyMuPDF: {convert_err}")
                
                try:
                    # Fallback to basic PyMuPDF for PDFs only
                    # Pass the path to the temporary file; fitz.open() can take a filepath directly
                    pdf_document = fitz.open(temp_file_path) 
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
                    detailed_error_msg = (
                        f"Failed to extract text with both pymupdf4llm and PyMuPDF. "
                        f"PyMuPDF fallback error: {fallback_err}. "
                        f"Original pymupdf4llm error: {convert_err}."
                    )
                    logging.error(f"[{material_id}] {detailed_error_msg}")
                    raise Exception(detailed_error_msg)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logging.info(f"[{material_id}] Successfully deleted temporary file: {temp_file_path}")
                except Exception as e_remove:
                    logging.error(f"[{material_id}] Error deleting temporary file {temp_file_path}: {e_remove}")
        
        # --- Process and Store Chunks ---
        if extracted_text:
            logging.info(f"[{material_id}] Extracted text length: {len(extracted_text)} characters")
            
            # Generate parent chunks (larger chunks for context)
            parent_splitter = MarkdownTextSplitter(
                chunk_size=PARENT_CHUNK_SIZE, 
                chunk_overlap=PARENT_CHUNK_OVERLAP
            )
            parent_chunks = parent_splitter.split_text(extracted_text)
            
            logging.critical(f"[{material_id}] WORKER PHASE 3: Creating semantic chunks with text splitters")
            logging.info(f"[{material_id}] Created {len(parent_chunks)} parent chunks")
            
            # Generate embeddings for all parent chunks
            
            logging.critical(f"[{material_id}] WORKER PHASE 4: Processing {len(parent_chunks_md)} parent chunks, generating child chunks and embeddings...")
            # Loop through parent chunks (Markdown)
            for i, parent_chunk_content in enumerate(parent_chunks_md):
                parent_chunk_id = str(uuid.uuid4())
                parent_record = {
                    "id": parent_chunk_id,
                    "material_id": material_id,
                    "content": parent_chunk_content,
                    "index": i
                }
                all_parent_chunk_records.append(parent_record)

                # Split parent chunk into smaller child chunks for embedding
                child_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHILD_CHUNK_SIZE,
                    chunk_overlap=CHILD_CHUNK_OVERLAP
                )
                child_chunks_texts = child_splitter.split_text(parent_chunk_content)
                
                if not child_chunks_texts:
                    logging.warning(f"[{material_id}] Parent chunk {i+1} (ID: {parent_chunk_id}) resulted in no child chunks after splitting. Skipping.")
                    continue
                
                logging.info(f"[{material_id}] Parent chunk {i+1} (ID: {parent_chunk_id}) split into {len(child_chunks_texts)} child chunks. Generating embeddings...")
                
                try:
                    # Generate embeddings for all child_chunks_texts in this parent_chunk in one batch
                    embedding_response = self.openai_client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=child_chunks_texts  # Pass the list of texts
                    )
                    
                    if len(embedding_response.data) != len(child_chunks_texts):
                        logging.error(f"[{material_id}] Mismatch in expected ({len(child_chunks_texts)}) and received ({len(embedding_response.data)}) embeddings for parent chunk {i+1} (ID: {parent_chunk_id}). Skipping child chunks for this parent.")
                        continue 

                    for j, child_text in enumerate(child_chunks_texts):
                        embedding = embedding_response.data[j].embedding
                        
                        child_record = {
                            "id": str(uuid.uuid4()),
                            "material_id": material_id,
                            "parent_chunk_id": parent_chunk_id,
                            "content": child_text,
                            "embedding": embedding,
                            "index_in_parent": j
                        }
                        all_child_chunk_records.append(child_record)
                except Exception as emb_ex:
                    logging.error(f"[{material_id}] Error generating embeddings for child chunks of parent chunk {i+1} (ID: {parent_chunk_id}): {str(emb_ex)}")
                    # Continue to next parent chunk if embeddings fail for current one's children
                    continue 
            
            # Store parent chunk records in Supabase
            if all_parent_chunk_records:
                logging.critical(f"[{material_id}] WORKER PHASE 5A: Storing {len(all_parent_chunk_records)} parent chunk records in Supabase table 'parent_chunks'...")
                batch_size = 50
                for k_idx in range(0, len(all_parent_chunk_records), batch_size):
                    batch = all_parent_chunk_records[k_idx:k_idx+batch_size]
                    try:
                        self.supabase_client.table('ParentChunk').insert(batch).execute()
                    except Exception as db_ex:
                        logging.error(f"[{material_id}] Error inserting batch of parent chunks: {str(db_ex)}")
                        raise # Re-raising to fail the task if DB insert fails critically

            # Store child chunk records (with embeddings) in Supabase
            if all_child_chunk_records:
                logging.critical(f"[{material_id}] WORKER PHASE 5B: Storing {len(all_child_chunk_records)} child chunk (embedding) records in Supabase table 'chunks'...")
                batch_size = 50
                for k_idx in range(0, len(all_child_chunk_records), batch_size):
                    batch = all_child_chunk_records[k_idx:k_idx+batch_size]
                    try:
                        self.supabase_client.table('chunks').insert(batch).execute()
                    except Exception as db_ex:
                        logging.error(f"[{material_id}] Error inserting batch of child chunks: {str(db_ex)}")
                        raise # Re-raising
            
            # --- Update Material Status ---
            logging.critical(f"[{material_id}] WORKER COMPLETE: Processing complete, updating material status to COMPLETE")
            self.supabase_client.table('materials').update({
                "status": 'COMPLETE',
                "errorMessage": None,
                "totalChunks": len(all_child_chunk_records)
            }).eq('id', material_id).execute()
            
            success = True
            logging.info(f"[{material_id}] Successfully processed material")
        else:
            error_message = "Failed to extract text from document"
            logging.error(f"[{material_id}] {error_message}")
            
    except Exception as e:
        error_message = f"Error processing material: {str(e)}"
        logging.critical(f"[{material_id}] WORKER ERROR: {error_message}")
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
