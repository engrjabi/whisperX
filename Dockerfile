FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# System dependencies for ffmpeg
RUN apt-get update && apt-get install -y ffmpeg git

# Python
RUN apt-get update && apt-get install -y python3 python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY api.py .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
