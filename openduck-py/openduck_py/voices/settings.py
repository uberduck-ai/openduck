import os
import torch

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ESPEAK_LANGUAGES = {
    "english": "en-us",
    "spanish": "es",
    "dutch": "nl",
    "german": "de",
    "japanese": "ja",
}

LATIN_CHARACTERS = r"[A-Za-z]"
PUNCTUATION_CHARACTERS = r"[\s\.,;:\'\"!? -]"
