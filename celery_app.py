import logging
import sys
from celery import Celery
from celery.signals import worker_ready, worker_init, task_prerun, task_postrun, task_failure, task_success
from app.core.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

# Configure logging to show worker events clearly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

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

# --- Celery Signal Handlers ---
@worker_init.connect
def worker_init_handler(sender, **kwargs):
    logging.critical(f"WORKER INIT: Celery worker initializing")
    logging.critical(f"WORKER INIT: Connected to broker: {CELERY_BROKER_URL}")

@worker_ready.connect
def worker_ready_handler(sender, **kwargs):
    logging.critical(f"WORKER READY: Celery worker is now ready for tasks. Hostname: {sender.hostname}")

@task_prerun.connect
def task_prerun_handler(sender, task_id, task, args, kwargs, **_):
    logging.critical(f"TASK START: {task.name}[{task_id}] starting execution with args: {args}")

@task_postrun.connect
def task_postrun_handler(sender, task_id, task, args, kwargs, **_):
    logging.critical(f"TASK END: {task.name}[{task_id}] finished execution")

@task_failure.connect
def task_failure_handler(sender, task_id, exception, args, kwargs, traceback, einfo, **_):
    logging.critical(f"TASK FAILURE: {sender.name}[{task_id}] failed: {exception}")

@task_success.connect
def task_success_handler(sender, result, **kwargs):
    logging.critical(f"TASK SUCCESS: {sender.name} completed with result: {result}")

if __name__ == "__main__":
    logging.critical("Starting Celery worker...")
    celery_app.start()
