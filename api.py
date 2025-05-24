from fastapi import FastAPI, File, UploadFile, Form
import whisperx
import torch
import os
from typing import Optional

app = FastAPI()

@app.post('/transcribe')
async def transcribe_audio(
    file: UploadFile = File(...),
    model_name: str = Form("tiny"),                    # ex: "tiny", "base", ..., "large-v2"
    language_code: str = Form("en"),                   # ex: "en", "es", ...
    output_format: Optional[str] = Form("segments")    # ex: "segments" or "full"
):
    # Save to temp file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, 'wb') as out_file:
        out_file.write(await file.read())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    compute_type = "float16" if device == "cuda" else "float32"

    # 1. Transcribe
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    audio = whisperx.load_audio(temp_path)
    result = model.transcribe(audio, batch_size=batch_size)
    del model

    # 2. Align
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    del model_a

    os.remove(temp_path)

    if output_format == "full":
        return result_aligned
    else:
        return {"segments": result_aligned["segments"]}
