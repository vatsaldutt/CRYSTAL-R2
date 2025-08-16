# my_tts_service.py
import aiohttp
from loguru import logger
from typing import AsyncGenerator

from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import (
    TTSStartedFrame, TTSStoppedFrame, TTSAudioRawFrame, ErrorFrame
)

class MyTTSServerService(TTSService):
    """Custom TTS service that calls your own FastAPI/WebSocket/HTTP server."""

    def __init__(self, base_url="http://localhost:8000/tts", sample_rate=24000, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.base_url = base_url

    async def run_tts(self, text: str) -> AsyncGenerator:
        """Send text to your TTS server and stream back audio frames."""
        logger.debug(f"MyTTS generating: {text}")
        yield TTSStartedFrame()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, json={"text": text}) as resp:
                    if resp.status != 200:
                        error_msg = await resp.text()
                        logger.error(f"TTS server error: {error_msg}")
                        yield ErrorFrame(error_msg)
                        return

                    # Assume your server returns raw PCM bytes
                    audio_bytes = await resp.read()

            yield TTSAudioRawFrame(
                audio=audio_bytes,
                sample_rate=self.sample_rate,
                num_channels=1
            )

        except Exception as e:
            logger.error(f"MyTTS exception: {e}")
            yield ErrorFrame(str(e))

        yield TTSStoppedFrame()
