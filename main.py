import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import check_config
from app.core.clients import init_clients

# Initialize FastAPI app
app = FastAPI(
    title="SageDesk RAG Processor",
    description="API for processing documents and generating embeddings for RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize clients on startup
@app.on_event("startup")
async def startup_event():
    # Check configuration
    if not check_config():
        logging.critical("Server cannot start due to missing critical configuration.")
        # We don't exit here because the application would still run, but it'll log the error
    
    # Initialize clients
    init_clients()
    logging.info("FastAPI application started successfully")

# Register routers
app.include_router(api_router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": f"An unexpected error occurred: {str(exc)}"}
    )

# Add a health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

# Entry point for running the application directly
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
