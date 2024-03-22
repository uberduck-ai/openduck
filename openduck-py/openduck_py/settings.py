import os

HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
EMB_MATCH_THRESHOLD = 0.5
WS_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
IS_DEV = bool(os.environ.get("IS_DEV"))
ML_API_URL = os.environ["ML_API_URL"]
# Set to 1024 for the esp32, but larger CHUNK_SIZE is needed to prevent choppiness with the local client
CHUNK_SIZE = 10240
LOG_TO_SLACK = bool(os.environ.get("LOG_TO_SLACK", False))
CHAT_MODEL = "azure/gpt-35-turbo-deployment"
CHAT_MODEL_GPT4 = "azure/gpt-4-deployment"
AUDIO_UPLOAD_BUCKET = os.environ.get("AUDIO_UPLOAD_BUCKET", "openduck-us-west-2")

# to not break existing env files
os.environ["AZURE_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY", "")
os.environ["AZURE_API_BASE"] = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
os.environ["AZURE_API_VERSION"] = "2023-05-15"
