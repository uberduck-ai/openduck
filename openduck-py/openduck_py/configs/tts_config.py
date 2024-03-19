from typing import Literal

TTSProviders = Literal["local", "elevenlabs", "openai"]


class TTSConfig:
    def __init__(self, provider: TTSProviders = "local"):
        self.provider = provider
