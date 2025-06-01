import logging
from celery import Celery
from app.core.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

# Configure Celery application
celery_app = Celery(
    "sagedesk_rag_processor",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.tasks.rag_processor"]  # Include task modules
)

# Optional Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,  # Helps with fair task distribution
    task_acks_late=True,  # Tasks acknowledged after execution (more reliable)
)

if __name__ == "__main__":
    logging.info("Starting Celery worker...")
    celery_app.start()
