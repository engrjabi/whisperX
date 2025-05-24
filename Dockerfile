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
#    Copy only the files that affect dependency resolution so this layer stays
#    cached when we edit source files.
# ------------------------------------------------------------
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# 4. Application source code
#    Copy the rest of the repo and install it in editable mode. Changes to the
#    code will rebuild only the following small layers, keeping the heavy
#    dependency layers cached.
# ------------------------------------------------------------
COPY . /app
RUN pip install --no-deps -e .

# ------------------------------------------------------------
# 5. Model files (kept in the image so first run is fast)
# ------------------------------------------------------------
RUN python download_models.py

# ------------------------------------------------------------
# 6. Expose & run
# ------------------------------------------------------------
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
