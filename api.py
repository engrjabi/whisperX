from fastapi import FastAPI, File, UploadFile, Form, Response
import whisperx
import torch
import os
import io
from typing import Optional
from whisperx.SubtitlesProcessor import SubtitlesProcessor, format_timestamp

app = FastAPI()

def segments_to_srt(segments):
    def format_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace('.', ',')
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        text = seg['text'].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)

@app.post('/transcribe')
async def transcribe_audio(
    file: UploadFile = File(...),
    model_name: str = Form("tiny"),
    language_code: str = Form("en"),
    output_format: str = Form("segments"),  # "segments" (default), "full", or "srt"
    batch_size: int = Form(16),
    compute_type: Optional[str] = Form(None),
    return_char_alignments: bool = Form(False),
    diarize: bool = Form(False),
    max_line_width: Optional[int] = Form(None),
    max_line_count: Optional[int] = Form(None),
):
    # Save to temp file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, 'wb') as out_file:
        out_file.write(await file.read())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "float32"

    # 1. Transcribe (pass through user-specified language to skip automatic language detection)
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language_code,
    )
    audio = whisperx.load_audio(temp_path)
    result = model.transcribe(audio, batch_size=batch_size)
    del model

    # 2. Align
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    result_aligned = whisperx.align(
        result["segments"], model_a, metadata, audio, device, return_char_alignments=return_char_alignments
    )
    del model_a

    os.remove(temp_path)

    if output_format == "srt":
        # Use advanced SubtitlesProcessor for cleaner sentence-aware splitting
        subtitles_processor = SubtitlesProcessor(
            segments=result_aligned["segments"],
            lang=language_code,
            max_line_length=max_line_width or 45,
            is_vtt=False,
        )

        subtitles = subtitles_processor.process_segments(advanced_splitting=True)

        buf = io.StringIO()
        for idx, sub in enumerate(subtitles, 1):
            start_ts = format_timestamp(sub["start"], is_vtt=False)
            end_ts = format_timestamp(sub["end"], is_vtt=False)
            text = sub["text"].strip()
            buf.write(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n\n")

        return Response(content=buf.getvalue(), media_type="text/plain")
    elif output_format == "full":
        return result_aligned
    else:
        return {"segments": result_aligned["segments"]}


