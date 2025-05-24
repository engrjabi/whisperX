from fastapi import FastAPI, File, UploadFile
import whisperx
import torch
import os

app = FastAPI()

@app.post('/transcribe')
async def transcribe_audio(file: UploadFile = File(...)):
    # Save to temp file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, 'wb') as out_file:
        out_file.write(await file.read())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    compute_type = "float16" if device == "cuda" else "float32"

    # 1. Transcribe
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(temp_path)
    result = model.transcribe(audio, batch_size=batch_size)
    del model

    # 2. Align
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    del model_a

    # Clean up temp file
    os.remove(temp_path)

    # Return result in plain JSON
    return {"segments": result["segments"]}
