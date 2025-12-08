# Dockerfile.api
FROM python:3.11-slim

# set workdir
WORKDIR /app

# copy code
COPY . /app

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# expose port
EXPOSE 8000

# use uvicorn directly (bind to 0.0.0.0)
CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
