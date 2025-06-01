web: gunicorn -k uvicorn.workers.UvicornWorker main:app --preload -b 0.0.0.0:$PORT
worker: celery -A celery_app.celery_app worker -l info --concurrency=2