import os

HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
EMB_MATCH_THRESHOLD = 0.5
WS_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
DEPLOY_ENV = os.environ.get("DEPLOY_ENV", "dev")
IS_DEV = DEPLOY_ENV == "dev"
CHUNK_SIZE = 1024  # NOTE (Sam): this esp32-specific parameter causes it to fail - 32768 is better on my dev box but still not perfect
LOG_TO_SLACK = bool(os.environ.get("LOG_TO_SLACK", False))
CHAT_MODEL = "azure/gpt-35-turbo-deployment"
UTILITY_MODEL = "azure/gpt-35-turbo-deployment"
