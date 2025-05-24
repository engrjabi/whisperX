import whisperx
import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    # Download the large-v3 model for high quality
    print("Downloading WhisperX 'large-v3' model...")
    whisperx.load_model("large-v3", device, compute_type=compute_type)

    # Download the align model (for English)
    print("Downloading alignment model for English...")
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
