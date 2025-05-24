FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# System dependencies for ffmpeg and python
RUN apt-get update && apt-get install -y ffmpeg git python3 python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install torch first to leverage Docker layer caching
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cu121

# Copy and install the rest of requirements (frequently updated dependencies)
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY api.py .
COPY download_models.py .

# Download the smallest WhisperX model and align model at build time
RUN python download_models.py

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
