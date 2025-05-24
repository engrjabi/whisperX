# download_models.py
import whisperx
import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    # Download the smallest model ('tiny') for fast iteration
    print("Downloading WhisperX 'tiny' model...")
    whisperx.load_model("tiny", device, compute_type=compute_type)

    # Download the align model (for English)
    print("Downloading alignment model for English...")
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
