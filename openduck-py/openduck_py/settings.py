import os
from typing import Literal

# Env settings and secrets
IS_DEV = bool(os.environ.get("IS_DEV"))
ML_API_URL = os.environ["ML_API_URL"]
LOG_TO_SLACK = bool(os.environ.get("LOG_TO_SLACK", False))
AUDIO_UPLOAD_BUCKET = os.environ.get("AUDIO_UPLOAD_BUCKET", "openduck-us-west-2")
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
DEEPGRAM_API_SECRET = os.environ.get("DEEPGRAM_API_SECRET")

# Audio settings
WS_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
# Set to 1024 for the esp32, but larger CHUNK_SIZE is needed to prevent choppiness with the local client
CHUNK_SIZE = 10240

EMB_MATCH_THRESHOLD = 0.5
LOG_TO_S3 = False
NO_SPEECH_PROB_THRESHOLD = 0.5

# LLM settings
ChatModels = Literal[
    "azure/gpt-35-turbo-deployment", "azure/gpt-4-deployment", "groq/mixtral-8x7b-32768"
]
CHAT_MODEL = "azure/gpt-35-turbo-deployment"
TEMPERATURE = 1.2
AUDIO_UPLOAD_BUCKET = os.environ.get("AUDIO_UPLOAD_BUCKET", "openduck-us-west-2")
RECORDING_UPLOAD_BUCKET = os.environ.get(
    "RECORDING_UPLOAD_BUCKET", "openduck-us-west-2"
)
LOG_TO_S3 = True

# ASR settings
ASRMethod = Literal["deepgram", "whisper"]
ASR_METHOD: ASRMethod = "whisper"

# to not break existing env files
os.environ["AZURE_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY", "")
os.environ["AZURE_API_BASE"] = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
os.environ["AZURE_API_VERSION"] = "2023-05-15"
