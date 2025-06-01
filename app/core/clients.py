import logging
import supabase
import openai
from app.core.config import (
    SUPABASE_URL, 
    SUPABASE_SERVICE_KEY, 
    OPENAI_API_KEY, 
    OPENAI_API_BASE, 
    OPENROUTER_API_KEY
)

# Initialize clients as None, will be setup in the init_clients function
supabase_client = None
openai_client = None

def init_clients():
    """Initialize Supabase and OpenAI clients"""
    global supabase_client, openai_client
    
    # Initialize Supabase client
    try:
        if SUPABASE_URL and SUPABASE_SERVICE_KEY:
            supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            logging.info("Supabase client initialized successfully.")
        else:
            logging.error("Supabase URL or Service Key missing. Supabase client not initialized.")
    except Exception as e:
        logging.exception(f"FATAL: Failed to initialize Supabase client: {e}")
    
    # Initialize OpenAI client
    try:
        if OPENAI_API_KEY:
            openai_client = openai.OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_API_BASE if OPENROUTER_API_KEY else None  # Set base_url only for OpenRouter
            )
            logging.info(f"OpenAI client initialized successfully (Using Base URL: {openai_client.base_url}).")
        else:
            logging.error("OpenAI API Key (or OpenRouter Key) missing. OpenAI client not initialized.")
    except Exception as e:
        logging.exception(f"FATAL: Failed to configure OpenAI client: {e}")
    
    return supabase_client, openai_client

def get_supabase_client():
    """Get the Supabase client, initializing if necessary"""
    global supabase_client
    if not supabase_client:
        init_clients()
    return supabase_client

def get_openai_client():
    """Get the OpenAI client, initializing if necessary"""
    global openai_client
    if not openai_client:
        init_clients()
    return openai_client
