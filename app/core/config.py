import os
import logging
import sys
from dotenv import load_dotenv

# Load environment variables from .env file for local dev
load_dotenv()

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- Configuration Variables ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
WEBHOOK_SECRET = os.environ.get("RAG_WEBHOOK_SECRET")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# Use OpenRouter endpoint if key is present, else default to OpenAI
OPENAI_API_BASE = "https://openrouter.ai/api/v1" if OPENROUTER_API_KEY else "https://api.openai.com/v1"
OPENAI_API_KEY = OPENROUTER_API_KEY or os.environ.get("OPENAI_API_KEY") # Use OpenRouter key first
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

# Chunking configuration
PARENT_CHUNK_SIZE = 2000 # Target size for Markdown chunks (adjust as needed)
PARENT_CHUNK_OVERLAP = 200 # Overlap for Markdown splitter (can be 0)
CHILD_CHUNK_SIZE = 400    # Target size for embedding chunks
CHILD_CHUNK_OVERLAP = 50  # Overlap for recursive splitter

# Celery configuration
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

def check_config():
    """Verifies that all required configuration is present"""
    config_ok = True
    
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logging.critical("FATAL ERROR: Missing Supabase environment variables (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY).")
        config_ok = False
    if not WEBHOOK_SECRET:
        logging.critical("FATAL ERROR: Missing RAG_WEBHOOK_SECRET environment variable.")
        config_ok = False
    if not OPENAI_API_KEY:
        logging.critical("FATAL ERROR: Missing embedding API key (OPENROUTER_API_KEY or OPENAI_API_KEY).")
        config_ok = False
    if not CELERY_BROKER_URL:
        logging.warning("Warning: CELERY_BROKER_URL not set. Using default Redis URL.")
    
    return config_ok
