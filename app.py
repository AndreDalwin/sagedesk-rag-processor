import os
import supabase
import pymupdf4llm
import pymupdf # Re-import pymupdf as it's needed for pymupdf.open()
import fitz      # Import fitz for PyMuPDF core functionalities and exceptions
import openai
from flask import Flask, request, jsonify
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import uuid
import logging # For better logging
import urllib.parse # Import for URL encoding and decoding
import sys # Import sys to access stdout
import io  # Import io for BytesIO stream
import threading # Import threading

load_dotenv() # Load environment variables from .env file for local dev

# --- Configuration ---
# Explicitly configure logging to use stdout
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
WEBHOOK_SECRET = os.environ.get("RAG_WEBHOOK_SECRET")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Use OpenRouter endpoint if key is present, else default to OpenAI
OPENAI_API_BASE = "https://openrouter.ai/api/v1" if OPENROUTER_API_KEY else "https://api.openai.com/v1"
OPENAI_API_KEY = OPENROUTER_API_KEY or os.environ.get("OPENAI_API_KEY") # Use OpenRouter key first
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")

PARENT_CHUNK_SIZE = 2000 # Target size for Markdown chunks (adjust as needed)
PARENT_CHUNK_OVERLAP = 200 # Overlap for Markdown splitter (can be 0)
CHILD_CHUNK_SIZE = 400    # Target size for embedding chunks
CHILD_CHUNK_OVERLAP = 50     # Overlap for recursive splitter

# --- Initialize Clients ---
app = Flask(__name__)
supabase_client = None
openai_client = None

try:
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logging.info("Supabase client initialized successfully.")
    else:
        logging.error("Supabase URL or Service Key missing. Supabase client not initialized.")
except Exception as e:
    logging.exception(f"FATAL: Failed to initialize Supabase client: {e}")

try:
    if OPENAI_API_KEY:
        # Correct way to initialize the OpenAI client v1.x
        openai_client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE if OPENROUTER_API_KEY else None # Set base_url only for OpenRouter
        )
        logging.info(f"OpenAI client initialized successfully (Using Base URL: {openai_client.base_url}).")
    else:
        logging.error("OpenAI API Key (or OpenRouter Key) missing. OpenAI client not initialized.")
except Exception as e:
     logging.exception(f"FATAL: Failed to configure OpenAI client: {e}")


# --- Helper Functions (Chunking) ---
# Using Langchain splitters below

# --- Background Processing Function ---
def _process_material_in_background(material_id, storage_path):
    """Handles the entire RAG processing pipeline for a single material."""
    # Ensure clients are available (might be redundant if checked before starting thread, but safe)
    if not supabase_client or not openai_client:
        logging.error(f"[{material_id}] Background task cannot run: Clients not initialized.")
        # Attempt to update status to FAILED even if clients weren't fully ready
        if supabase_client and material_id:
             try:
                 supabase_client.table('materials').update({
                     "status": 'FAILED',
                     "errorMessage": "Server configuration error (clients unavailable)"
                 }).eq('id', material_id).execute()
             except Exception as db_err:
                 logging.exception(f"[{material_id}] CRITICAL: Failed to update status to FAILED (clients unavailable).")
        return # Exit the thread

    # Instantiate MarkItDown converter here or globally if preferred
    # Instantiating here ensures it's fresh for each task but adds slight overhead.
    md_converter = MarkItDown(enable_plugins=False) # Disable plugins unless needed

    logging.info(f"[{material_id}] Background processing thread started.")
    try:
        # --- RAG Processing ---
        logging.info(f"[{material_id}] Step 1: Updating status to PROCESSING...")
        update_response = supabase_client.table('materials').update({"status": 'PROCESSING', "errorMessage": None}).eq('id', material_id).execute()
        if hasattr(update_response, 'error') and update_response.error:
             raise Exception(f"Failed update status to PROCESSING: {update_response.error}")


        # Step 2: Get Material details (already have from payload)
        logging.info(f"[{material_id}] Step 2: Details: path={storage_path}") # Simplified log

        # Step 3: Download file from Storage
        # Path from DB might contain URL-encoded characters (e.g., %20 for space)
        raw_storage_path = storage_path
        logging.info(f"[{material_id}] Step 3: Received storage path: {raw_storage_path}")
        decoded_storage_path = raw_storage_path # Initialize with raw path as fallback
        file_extension = "" # Initialize file extension
        try:
            # URL-decode the path to handle potential %20, etc.
            decoded_storage_path = urllib.parse.unquote(raw_storage_path)
            # Extract file extension
            _, file_extension = os.path.splitext(decoded_storage_path)
            file_extension = file_extension.lower() # Normalize to lowercase
            logging.info(f"[{material_id}] URL-decoded storage path: {decoded_storage_path}, Detected extension: {file_extension}")
        except Exception as e:
            logging.warning(f"[{material_id}] URL decoding/extension extraction failed for path: {raw_storage_path}. Error: {e}")
            # Attempt to get extension from raw path as fallback
            _, file_extension = os.path.splitext(raw_storage_path)
            file_extension = file_extension.lower()
            logging.info(f"[{material_id}] Using raw storage path: {raw_storage_path}, Fallback extension: {file_extension}")

        # Download to memory using the decoded path
        logging.info(f"[{material_id}] Attempting download with path: {decoded_storage_path}")
        file_response = supabase_client.storage.from_('materials').download(decoded_storage_path)
        if not file_response: # Check if download returned bytes
             # Check the actual response status if possible, the exception might be misleading
             raise Exception(f"Failed to download file from storage (empty response/permissions?). Path tried: {decoded_storage_path}")

        file_content_bytes = file_response
        logging.info(f"[{material_id}] Successfully downloaded file. Size: {len(file_content_bytes)} bytes")

        # Step 4: Extract content to Markdown
        logging.info(f"[{material_id}] Step 4: Extracting content to Markdown...")
        md_text = "" # Initialize md_text
        doc = None   # Initialize doc variable
        # all_pages_md = [] # List to store markdown from each page - Moved inside PDF block

        try:
            file_stream = io.BytesIO(file_content_bytes)

            if file_extension == ".pdf":
                logging.info(f"[{material_id}] Processing as PDF using pymupdf4llm...")
                all_pages_md = [] # List to store markdown from each page
                try:
                    # 1. Open the PDF bytes as a stream using pymupdf.open()
                    # pdf_stream = io.BytesIO(file_content_bytes) # Already created as file_stream
                    # Explicitly provide filetype hint for stream
                    doc = pymupdf.open(stream=file_stream, filetype="pdf")

                    # 2. Iterate through pages and extract markdown individually
                    logging.info(f"[{material_id}] Processing {doc.page_count} PDF pages...")
                    for page_num in range(doc.page_count):
                        page = doc.load_page(page_num) # Load the page
                        logging.info(f"[{material_id}] Attempting extraction for PDF page {page_num + 1}/{doc.page_count}...")
                        try:
                            # Extract markdown for the current page only
                            # Pass the doc object but specify the single page number (0-based)
                            page_md = pymupdf4llm.to_markdown(
                                doc,
                                pages=[page_num], # Process only this page
                                ignore_images=True,
                                ignore_graphics=True
                            )
                            # Basic sanitization (remove null bytes)
                            sanitized_page_md = page_md.replace('\x00', '')
                            if sanitized_page_md.strip(): # Add if not empty
                                all_pages_md.append(sanitized_page_md)
                                logging.info(f"[{material_id}] Successfully extracted PDF page {page_num + 1}. Length: {len(sanitized_page_md)}")
                            else:
                                logging.info(f"[{material_id}] PDF Page {page_num + 1} resulted in empty markdown after sanitization.")

                        except ValueError as ve:
                            # Catch the specific error for this page
                            if "not a textpage of this page" in str(ve):
                                logging.warning(f"[{material_id}] Skipping PDF page {page_num + 1} due to 'not a textpage' error: {ve}")
                            else:
                                # Re-raise other ValueErrors if needed, or log as warning
                                logging.warning(f"[{material_id}] Skipping PDF page {page_num + 1} due to other ValueError: {ve}")
                        except Exception as page_err:
                            # Catch other potential errors during single-page processing
                            logging.warning(f"[{material_id}] Skipping PDF page {page_num + 1} due to unexpected error: {page_err}")

                    # 3. Combine markdown from all successful pages
                    md_text = "\n\n---\n\n".join(all_pages_md) # Join with a separator
                    if not md_text:
                         logging.warning(f"[{material_id}] PDF Markdown extraction resulted in empty content after processing all pages.")
                         # Consider if this should be treated as an error or success with no content
                         # For now, we proceed, but splitting might yield nothing.

                    logging.info(f"[{material_id}] Finished PDF page-by-page extraction. Total Markdown Length: {len(md_text)}")

                # Catch specific RuntimeError related to passwords, if fitz.PasswordError isn't available
                # This error would likely occur during pymupdf.open(), before the page loop
                except RuntimeError as e:
                    if "password" in str(e).lower():
                         logging.warning(f"[{material_id}] Caught RuntimeError likely indicating PDF password protection during doc open: {e}")
                         raise Exception("PDF file requires a password.") from e
                    else:
                         # Re-raise other RuntimeErrors
                         logging.error(f"[{material_id}] Caught unexpected RuntimeError during PDF open/initial processing: {e}")
                         raise e # Re-raise if it's not the password error
                except Exception as e:
                    # Log the specific error during extraction (likely during open or initial doc processing)
                    logging.exception(f"[{material_id}] Failed during initial PDF document processing or non-page-specific extraction step.")
                    # Ensure original exception type/message is preserved if possible
                    raise Exception(f"Failed during PDF document processing step: {type(e).__name__}: {e}") from e
                finally:
                    # Ensure the document is closed if it was opened
                    if doc:
                        try:
                            doc.close()
                            logging.info(f"[{material_id}] Closed PyMuPDF document.")
                        except Exception as close_err:
                            logging.warning(f"[{material_id}] Error closing PyMuPDF document: {close_err}")

            elif file_extension == ".txt":
                logging.info(f"[{material_id}] Processing as TXT by decoding content...")
                try:
                    # Try decoding directly using UTF-8
                    md_text = file_content_bytes.decode('utf-8')
                    logging.info(f"[{material_id}] Successfully decoded TXT as UTF-8. Content Length: {len(md_text)}")
                except UnicodeDecodeError:
                    logging.warning(f"[{material_id}] Failed to decode TXT as UTF-8. Attempting fallback to latin-1...")
                    try:
                        # Fallback to latin-1
                        md_text = file_content_bytes.decode('latin-1')
                        logging.info(f"[{material_id}] Successfully decoded TXT using fallback latin-1. Content Length: {len(md_text)}")
                    except Exception as decode_err_fallback:
                        # This fallback is less likely to fail, but catch just in case
                        logging.exception(f"[{material_id}] Failed to decode TXT even with latin-1 fallback.")
                        raise Exception("Failed to decode TXT content with UTF-8 or latin-1.") from decode_err_fallback
                except Exception as e:
                    # Catch other potential errors during decode
                    logging.exception(f"[{material_id}] Unexpected error during TXT decoding.")
                    raise Exception(f"Failed during TXT decoding: {type(e).__name__}: {e}") from e

                # Basic sanitization (remove null bytes) - apply after successful decode
                md_text = md_text.replace('\x00', '')
                if not md_text.strip():
                    logging.warning(f"[{material_id}] Decoded TXT content is empty or whitespace-only.")

            else:
                # Handle unsupported file types
                logging.warning(f"[{material_id}] Unsupported file type: '{file_extension}'. Cannot extract content.")
                # Option 1: Fail the job
                raise Exception(f"Unsupported file type: '{file_extension}'")
                # Option 2: Mark as completed with a message (if you want to allow unsupported types)
                # supabase_client.table('materials').update({"status": 'COMPLETED', "errorMessage": f"Unsupported file type: {file_extension}"}).eq('id', material_id).execute()
                # logging.info(f"[{material_id}] Background thread finished: Unsupported file type.")
                # return # Exit thread

        except Exception as extraction_err:
             # This catches errors from the specific handlers or the outer try if needed
             logging.exception(f"[{material_id}] Step 4 failed: Content extraction error.")
             # Re-raise the caught exception to be handled by the main try-except block
             raise extraction_err


        # Step 5: Split into Parent and Child Chunks (using Langchain)
        logging.info(f"[{material_id}] Step 5: Splitting Markdown into Parent and Child chunks...")

        # 5a: Split by Markdown structure (Parent Chunks)
        # Adjust separators based on pymupdf4llm output if needed
        md_splitter = MarkdownTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)
        parent_chunks_docs = md_splitter.create_documents([md_text]) # Returns Document objects
        parent_chunk_contents = [doc.page_content for doc in parent_chunks_docs if doc.page_content.strip()]
        logging.info(f"[{material_id}] Split into {len(parent_chunk_contents)} potential Parent chunks (Markdown-based).")

        if not parent_chunk_contents:
             logging.warning(f"[{material_id}] No parent chunks generated after Markdown splitting. Input length was {len(md_text)}. Setting status to COMPLETED.")
             # Update status to COMPLETED (or maybe FAILED if no content is unexpected)
             supabase_client.table('materials').update({"status": 'COMPLETED', "errorMessage": "No content extracted or chunked"}).eq('id', material_id).execute()
             # return jsonify({"success": True, "materialId": material_id, "message": "No content to process"}), 200 # Cannot return JSON from thread
             logging.info(f"[{material_id}] Background thread finished: No content.")
             return # Exit thread


        parent_chunk_records_to_insert = []
        child_chunk_data_for_embedding = [] # List of tuples: (parent_chunk_id, child_content)

        # 5b: Split Parent Chunks into smaller Child Chunks
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
            # Correctly formatted list of separators
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=len
        )

        for parent_content in parent_chunk_contents:
            parent_chunk_id = str(uuid.uuid4())
            # Ensure parent content doesn't have null bytes either
            # Use replace with null character, not space
            sanitized_parent_content = parent_content.replace('\x00', '')
            parent_chunk_records_to_insert.append({
                "id": parent_chunk_id,
                "materialId": material_id,
                "content": sanitized_parent_content
            })

            # Split this parent chunk into child chunks
            child_splits = child_splitter.split_text(sanitized_parent_content)
            for child_content in child_splits:
                trimmed_child_content = child_content.strip()
                if trimmed_child_content: # Ensure not just whitespace
                     # Basic check for null bytes again, just in case
                     # Use replace with null character, not space
                    sanitized_child_content = trimmed_child_content.replace('\x00', '')
                    child_chunk_data_for_embedding.append((parent_chunk_id, sanitized_child_content))

        logging.info(f"[{material_id}] Generated {len(child_chunk_data_for_embedding)} Child chunks to embed.")

        # Step 6: Store Parent Chunks
        logging.info(f"[{material_id}] Step 6: Storing {len(parent_chunk_records_to_insert)} Parent chunks in DB...")
        if parent_chunk_records_to_insert:
             # Batch insert (Supabase Python client handles list directly)
             # Ensure correct table name 'ParentChunk' from schema
             response = supabase_client.table('ParentChunk').insert(parent_chunk_records_to_insert).execute()
             # Proper error checking for supabase-py v1/v2
             if hasattr(response, 'error') and response.error:
                logging.error(f"[{material_id}] Error inserting ParentChunk batch: {response.error}")
                raise Exception(f"Failed to store Parent Chunks: {response.error}")
             elif hasattr(response, 'data') and not response.data:
                 # Handle cases where insert might succeed but return no data unexpectedly
                 # Check response status if available, or look for specific error types
                 logging.warning(f"[{material_id}] ParentChunk insert executed but returned no data. Check Supabase logs.")
                 # Consider this potentially problematic, depending on Supabase client version behavior
             logging.info(f"[{material_id}] Successfully stored Parent chunks.")
        else:
             logging.info(f"[{material_id}] No Parent chunks to store.")


        # Step 7: Generate Embeddings for Child Chunks
        logging.info(f"[{material_id}] Step 7: Generating embeddings for {len(child_chunk_data_for_embedding)} Child chunks...")
        embeddings_data = [] # List of dicts: {parent_id, content, embedding}
        failed_embedding_count = 0

        # --- Batching Embedding Requests ---
        BATCH_SIZE_EMBEDDINGS = 50 # Adjust based on API limits and typical chunk sizes
        child_chunks_content_only = [content for _, content in child_chunk_data_for_embedding]

        for i in range(0, len(child_chunks_content_only), BATCH_SIZE_EMBEDDINGS):
             batch_content = child_chunks_content_only[i:i + BATCH_SIZE_EMBEDDINGS]
             batch_parent_ids = [pid for pid, _ in child_chunk_data_for_embedding[i:i + BATCH_SIZE_EMBEDDINGS]]
             logging.info(f"[{material_id}] Requesting embeddings for batch {i // BATCH_SIZE_EMBEDDINGS + 1} (size: {len(batch_content)})...")

             try:
                 # Use openai client configured for OpenRouter or OpenAI
                 emb_response = openai_client.embeddings.create(
                     input=batch_content, # Send batch
                     model=EMBEDDING_MODEL
                 )

                 if not emb_response.data or len(emb_response.data) != len(batch_content):
                      raise Exception(f"API returned unexpected number of embeddings. Expected {len(batch_content)}, got {len(emb_response.data) if emb_response.data else 0}")

                 for idx, embedding_item in enumerate(emb_response.data):
                     embedding = embedding_item.embedding
                     if not embedding:
                         # This case might indicate an issue with a specific item in the batch
                         logging.warning(f"[{material_id}] API returned no embedding vector for item {idx} in batch starting at {i}.")
                         failed_embedding_count += 1
                         continue # Skip this item

                     original_index = i + idx
                     parent_id = child_chunk_data_for_embedding[original_index][0]
                     original_content = child_chunk_data_for_embedding[original_index][1]

                     embeddings_data.append({
                         "parent_id": parent_id,
                         "content": original_content,
                         "embedding": embedding
                     })

             except Exception as e:
                 failed_embedding_count += len(batch_content) # Assume whole batch failed on error
                 logging.exception(f"[{material_id}] Warning: Failed to generate embeddings for batch starting at index {i}. Error: {e}")
                 # Optionally implement retry logic for the batch here

        logging.info(f"[{material_id}] Successfully generated {len(embeddings_data)} embeddings.")
        if failed_embedding_count > 0:
            logging.warning(f"[{material_id}] {failed_embedding_count} embedding requests failed.")
            # Decide if this constitutes a failure (e.g., if failed_embedding_count > len(child_chunk_data_for_embedding) * 0.5)
            # Example: Raise error if > 50% fail
            # if failed_embedding_count / len(child_chunk_data_for_embedding) > 0.5:
            #    raise Exception("High rate of embedding failures.")


        # Step 8: Store Child Chunks and Embeddings
        logging.info(f"[{material_id}] Step 8: Storing {len(embeddings_data)} Child chunks and embeddings in DB...")
        child_chunk_records_to_insert = []
        for data in embeddings_data:
            child_chunk_records_to_insert.append({
                "id": str(uuid.uuid4()),
                "materialId": material_id,
                "parentId": data["parent_id"],
                "content": data["content"],
                "embedding": data["embedding"],
                "metadata": {"parentId": data["parent_id"]} # Add metadata if your schema uses it
            })

        if child_chunk_records_to_insert:
             # Batch insert
             # Ensure correct table name 'chunks' from schema
             BATCH_SIZE_DB = 100 # Adjust as needed
             for i in range(0, len(child_chunk_records_to_insert), BATCH_SIZE_DB):
                 batch_data = child_chunk_records_to_insert[i:i + BATCH_SIZE_DB]
                 logging.info(f"[{material_id}] Inserting ChildChunk batch {i // BATCH_SIZE_DB + 1} (size: {len(batch_data)})")
                 response = supabase_client.table('chunks').insert(batch_data).execute()
                 if hasattr(response, 'error') and response.error:
                      logging.error(f"[{material_id}] Error inserting ChildChunk batch: {response.error}")
                      raise Exception(f"Failed to store Child Chunks batch: {response.error}")
                 elif hasattr(response, 'data') and not response.data:
                      logging.warning(f"[{material_id}] ChildChunk insert batch executed but returned no data.")

             logging.info(f"[{material_id}] Successfully stored {len(child_chunk_records_to_insert)} Child chunks.")
        else:
             logging.info(f"[{material_id}] No Child chunks with embeddings to store.")

        # Step 9: Update Material status to COMPLETED
        logging.info(f"[{material_id}] Step 9: Updating status to COMPLETED...")
        # Ensure correct table name 'materials'
        update_response = supabase_client.table('materials').update({"status": 'COMPLETED', "errorMessage": None}).eq('id', material_id).execute()
        if hasattr(update_response, 'error') and update_response.error:
             # Log the error but don't necessarily fail the whole request here, as processing is done
             logging.error(f"[{material_id}] Error updating status to COMPLETED: {update_response.error}")
             # Maybe return a different success message or status?
        else:
            logging.info(f"[{material_id}] Successfully updated status to COMPLETED.")


        logging.info(f"[{material_id}] Background processing thread finished successfully.")
        # No return value needed from thread

    except Exception as e:
        error_message = f"Error processing Material {material_id}: {str(e)}"
        logging.exception(f"[{material_id}] Background processing thread failed.") # Log detailed traceback

        if material_id and supabase_client:
            try:
                logging.info(f"[{material_id}] Attempting DB update for FAILED status in background thread.")
                supabase_client.table('materials').update({
                    "status": 'FAILED',
                    "errorMessage": error_message[:1000] # Truncate error
                }).eq('id', material_id).execute()
                logging.info(f"[{material_id}] Successfully updated status to FAILED in DB (from background thread).")
            except Exception as db_err:
                logging.exception(f"[{material_id}] CRITICAL: Failed to update status to FAILED after processing error in background thread.")

        # No return value needed from thread, error is logged and status updated

# --- Webhook Endpoint ---
@app.route('/process', methods=['POST'])
def process_webhook():
    if not supabase_client:
         logging.error("Webhook request received but Supabase client is not available.")
         return jsonify({"error": "Server not configured (Supabase client unavailable)"}), 500
    if not openai_client:
         logging.error("Webhook request received but OpenAI client is not available.")
         return jsonify({"error": "Server not configured (Embedding client unavailable)"}), 500

    # 1. Validate Secret
    incoming_secret = request.headers.get('X-Webhook-Secret')
    if not WEBHOOK_SECRET:
        logging.error("Webhook secret not configured on the server.")
        return jsonify({"error": "Server configuration error"}), 500
    if incoming_secret != WEBHOOK_SECRET:
        logging.warning("Unauthorized webhook attempt received.")
        return jsonify({"error": "Unauthorized"}), 401

    # 2. Parse Payload
    payload = request.json
    if not payload:
        logging.error("Received empty payload.")
        return jsonify({"error": "Invalid payload"}), 400

    event_type = payload.get('type')
    record = payload.get('record', {})
    material_id = record.get('id')
    status = record.get('status')

    # Determine if we should process
    should_process = False
    event_description = ""
    if event_type == 'INSERT' and material_id:
        should_process = True
        event_description = f"INSERT event for Material ID: {material_id}"
    elif event_type == 'UPDATE' and material_id and status == 'PENDING':
         should_process = True
         event_description = f"UPDATE event (Retry) for Material ID: {material_id}"

    if not should_process:
        logging.info(f"Ignoring event: Type={event_type}, ID={material_id}, Status={status}")
        return jsonify({"message": "Ignoring event"}), 200

    logging.info(f"Processing Request Accepted: {event_description}") # Log acceptance

    storage_path = record.get('storagePath')
    # file_type = record.get('fileType') # May not be needed if pymupdf4llm handles it

    if not material_id or not storage_path:
        logging.error(f"Error: Missing materialId or storagePath in payload for event: {event_description}")
        # Don't start a thread if essential info is missing
        return jsonify({"error": "Missing materialId or storagePath"}), 400

    try:
        # --- Start Background Processing ---
        logging.info(f"[{material_id}] Starting background thread for processing...")
        thread = threading.Thread(
            target=_process_material_in_background,
            args=(material_id, storage_path)
        )
        thread.daemon = True # Allows main program to exit even if threads are running (optional)
        thread.start()

        # Return immediately after starting the thread
        logging.info(f"[{material_id}] Webhook request acknowledged, processing started in background.")
        return jsonify({"message": "Processing started", "materialId": material_id}), 202 # HTTP 202 Accepted

    except Exception as e:
        # This catches errors during thread *creation/start*, not errors *within* the thread
        error_message = f"Error initiating processing for Material {material_id}: {str(e)}"
        logging.exception(f"[{material_id}] Failed to start background processing thread.")

        # Attempt to update status to FAILED even if thread didn't start properly
        if material_id and supabase_client:
            try:
                logging.info(f"[{material_id}] Attempting DB update for FAILED status (thread start failed).")
                supabase_client.table('materials').update({
                    "status": 'FAILED',
                    "errorMessage": f"Failed to start processing thread: {error_message[:900]}" # Truncate
                }).eq('id', material_id).execute()
                logging.info(f"[{material_id}] Successfully updated status to FAILED in DB (thread start failed).")
            except Exception as db_err:
                 logging.exception(f"[{material_id}] CRITICAL: Failed to update status to FAILED after thread start error.")

        return jsonify({"error": error_message}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Check essential configurations before starting
    CONFIG_OK = True
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
         logging.critical("FATAL ERROR: Missing Supabase environment variables (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY).")
         CONFIG_OK = False
    if not WEBHOOK_SECRET:
         logging.critical("FATAL ERROR: Missing RAG_WEBHOOK_SECRET environment variable.")
         CONFIG_OK = False
    if not OPENAI_API_KEY: # Check the combined key variable
         logging.critical("FATAL ERROR: Missing embedding API key (OPENROUTER_API_KEY or OPENAI_API_KEY).")
         CONFIG_OK = False

    if CONFIG_OK:
         logging.info("Starting Flask server...")
         # Port 5000 is common, Railway injects PORT env var
         port = int(os.environ.get('PORT', 5000))
         # Use gunicorn for production via Procfile, app.run is for local dev
         app.run(host='0.0.0.0', port=port, debug=False) # Set debug=False for production/Railway
    else:
        logging.critical("Server cannot start due to missing critical configuration.")
        # Optionally, sys.exit(1) here 