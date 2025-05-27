# WhisperX API Dockerized (GPU)

A simple, production-ready HTTP API for [whisperX](https://github.com/m-bain/whisperX) (speech-to-text + word alignment). GPU-only (CUDA/torch) by default.

## Features

- FastAPI REST API
- Transcribes and aligns audio
- Pass custom initial prompt/context for better results (`initial_prompt` parameter)
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

#### Minimal Example
```bash
curl -X POST "http://localhost:8000/transcribe" \
     -F "file=@your_audio.mp3" \
     -F "initial_prompt=This is a sample meeting about AI research."
```
You can provide a custom context or prompt for the first window of audio using `initial_prompt`:
- (e.g. vocabulary, topic, or important context)

**Note:** The prompt (combined with the model's output) is subject to the Whisper model context window, which is typically limited to 448 tokens for most models. If your prompt is too long, it will be automatically truncated.

#### Full/Verbose Example
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: text/plain" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/home/Documents/mb1ij9x3rn5qnbncb_output_v2.mp3" \
  -F "model_name=large-v3" \
  -F "language_code=tl" \
  -F "output_format=srt" \
  -F "batch_size=5" \
  -F "compute_type=float16" \
  -F "return_char_alignments=true" \
  -F "diarize=false" \
  -F "max_line_width=50" \
  -F "max_line_count=2" \
  -F "initial_prompt=This is a sample meeting about AI research."
```

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
