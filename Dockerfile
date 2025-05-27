FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ------------------------------------------------------------
# 1. System setup
# ------------------------------------------------------------
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 2. Python base dependencies
#    Install CUDA-enabled torch first – this layer rarely changes and can be
#    cached independently of application code.
# ------------------------------------------------------------
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------
# 3. Third-party Python packages (excluding our local code)
#    Install project dependencies before copying full source for better caching
# ------------------------------------------------------------
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# 4. Model download step
#    Copy only files required for model download first so we maximize caching
# ------------------------------------------------------------
COPY download_models.py /app/download_models.py
COPY whisperx /app/whisperx
RUN python download_models.py

# ------------------------------------------------------------
# 5. Application code and documentation
#    These changes do *not* bust model download cache
# ------------------------------------------------------------
COPY . /app
RUN pip install --no-deps -e .

# ------------------------------------------------------------
# 6. Expose & run
# ------------------------------------------------------------
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
