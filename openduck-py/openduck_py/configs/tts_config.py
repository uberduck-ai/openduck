from typing import Literal, Optional

TTSProviders = Literal["local", "azure", "elevenlabs", "openai"]


class TTSConfig:
    def __init__(
        self, provider: TTSProviders = "local", voice_id: Optional[str] = None
    ):
        self.provider = provider
        self.voice_id = voice_id
