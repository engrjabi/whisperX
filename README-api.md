# WhisperX API Dockerized (GPU)

A simple, production-ready HTTP API for [whisperX](https://github.com/m-bain/whisperX) (speech-to-text + word alignment). GPU-only (CUDA/torch) by default.

## Features

- FastAPI REST API
- Transcribes and aligns audio
- **No HuggingFace or diarization** (simpler, no extra dependencies)
- GPU-accelerated (CUDA)

---

## Quick Start

### 1. Build

```bash
docker build -t whisperx-api .
```

### 2. Run (with GPU)

```bash
docker run --gpus all -p 8000:8000 whisperx-api
```

Requires [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/user-guide.html) for GPU access.

### 3. Usage Example

```bash
curl -X POST "http://localhost:8000/transcribe" \
     -F "file=@your_audio.mp3"
```

**Returns:** JSON with aligned segments.

---

## Files

- `api.py`: FastAPI app – POST `/transcribe` for audio transcription/alignment
- `Dockerfile`: CUDA-enabled docker build
- `requirements.txt`: All dependencies
- `README-api.md`: This doc

---

## Notes

- The endpoint expects an audio file via HTTP form upload (not raw body).
- Large models may require significant GPU VRAM (see `batch_size` parameter in `api.py`).
- Curious about further features? PRs/issues welcome!

---

MIT License
