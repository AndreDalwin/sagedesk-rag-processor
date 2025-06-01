import logging
from fastapi import APIRouter, Request, Response, HTTPException, Depends, BackgroundTasks, status
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.core.clients import get_supabase_client
from app.core.config import WEBHOOK_SECRET
from app.tasks.rag_processor import process_material_task

# Create router
router = APIRouter()

# Define response and request models
class WebhookResponse(BaseModel):
    message: str
    materialId: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str

@router.post("/process", 
             response_model=WebhookResponse, 
             responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
             status_code=status.HTTP_202_ACCEPTED)
async def process_webhook(request: Request):
    """
    Webhook endpoint for Supabase to trigger document processing.
    Validates the webhook secret and starts a background Celery task to process the document.
    """
    # Verify Webhook Secret (if configured)
    secret_header = request.headers.get("X-Webhook-Secret")
    if WEBHOOK_SECRET and secret_header != WEBHOOK_SECRET:
        logging.warning(f"Invalid webhook secret received: {secret_header}")
        raise HTTPException(status_code=401, detail="Invalid webhook secret")
    
    # Get the JSON payload
    try:
        payload = await request.json()
    except Exception as e:
        logging.error(f"Error parsing JSON payload: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    # Extract record data
    record = payload.get("record", {})
    material_id = record.get("id")
    status_value = record.get("status")
    event_type = payload.get("type")
    
    # Determine if we should process this webhook
    should_process = False
    event_description = ""
    if event_type == 'INSERT' and material_id:
        should_process = True
        event_description = f"INSERT event for Material ID: {material_id}"
    elif event_type == 'UPDATE' and material_id and status_value == 'PENDING':
        should_process = True
        event_description = f"UPDATE event (Retry) for Material ID: {material_id}"

    if not should_process:
        logging.info(f"Ignoring event: Type={event_type}, ID={material_id}, Status={status_value}")
        return WebhookResponse(message="Ignoring event")

    logging.info(f"Processing Request Accepted: {event_description}")

    storage_path = record.get('storagePath')

    if not material_id or not storage_path:
        logging.error(f"Error: Missing materialId or storagePath in payload for event: {event_description}")
        # Don't start processing if essential info is missing
        raise HTTPException(status_code=400, detail="Missing materialId or storagePath")

    try:
        # --- Start Celery Task ---
        logging.info(f"[{material_id}] Dispatching Celery task for processing...")
        process_material_task.delay(material_id, storage_path)

        # Return immediately after starting the task
        logging.info(f"[{material_id}] Webhook request acknowledged, processing started in Celery worker.")
        return WebhookResponse(message="Processing started", materialId=material_id)

    except Exception as e:
        # This catches errors during task creation
        error_message = f"Error initiating processing for Material {material_id}: {str(e)}"
        logging.exception(f"[{material_id}] Failed to start Celery task.")

        # Attempt to update status to FAILED if task couldn't be dispatched
        supabase_client = get_supabase_client()
        if material_id and supabase_client:
            try:
                logging.info(f"[{material_id}] Attempting DB update for FAILED status (task dispatch failed).")
                supabase_client.table('materials').update({
                    "status": 'FAILED',
                    "errorMessage": f"Failed to start processing task: {error_message[:900]}"  # Truncate
                }).eq('id', material_id).execute()
                logging.info(f"[{material_id}] Successfully updated status to FAILED in DB (task dispatch failed).")
            except Exception as db_err:
                logging.exception(f"[{material_id}] CRITICAL: Failed to update status to FAILED after task dispatch error.")

        raise HTTPException(status_code=500, detail=error_message)
