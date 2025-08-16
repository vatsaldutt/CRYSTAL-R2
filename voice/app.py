# app.py
import os
import asyncio
import platform
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import TextFrame, EndFrame, AudioRawFrame
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams
from pipecat.transcriptions.language import Language

# Config
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_TOKEN = os.getenv("LIVEKIT_TOKEN")
ROOM_NAME = os.getenv("LIVEKIT_ROOM", "demo-room")
BOT_NAME = os.getenv("BOT_NAME", "PipecatMacBot")
TTS_WS_URL = os.getenv("TTS_WS_URL", "ws://localhost:8000/ws/tts")
USE_ULTRAVOX = os.getenv("USE_ULTRAVOX", "0") == "1"
HF_TOKEN = os.getenv("HF_TOKEN", None)

# --- Custom TTS service that calls your realtime websocket TTS ---
import websockets


class TTSWebsocketService(FrameProcessor):
    """
    Pipecat-style FrameProcessor that consumes TextFrame frames
    and pushes AudioRawFrame frames produced by an external TTS websocket.
    """

    def __init__(self, tts_ws_url: str, chunk_size: int = 50):
        super().__init__()
        self.tts_ws_url = tts_ws_url
        self.chunk_size = chunk_size

    async def process_frame(self, frame, direction=None):
        # Handle TextFrame → synthesize audio
        if isinstance(frame, TextFrame):
            text = getattr(frame, "text", None) or getattr(frame, "content", "")
            if not text:
                return
            async for audio_frame in self._synthesize_stream(text):
                await self.push_frame(audio_frame)

        # Pass through EndFrame so pipeline knows to finish
        elif isinstance(frame, EndFrame):
            await self.push_frame(frame)

    async def _synthesize_stream(self, text):
        """
        Connect to websocket TTS server, send init JSON,
        and yield AudioRawFrame objects as we receive binary audio chunks.
        """
        sr = None
        channels = 1
        try:
            async with websockets.connect(self.tts_ws_url, max_size=None) as ws:
                # send init JSON
                init = {"text": text, "chunk_size": self.chunk_size}
                await ws.send(json.dumps(init))

                while True:
                    msg = await ws.recv()

                    if isinstance(msg, (bytes, bytearray)):
                        # Binary audio chunk (float32)
                        arr = np.frombuffer(msg, dtype=np.float32)
                        if channels > 1:
                            arr = arr.reshape(-1, channels).T  # (channels, frames)
                        else:
                            arr = np.expand_dims(arr, 0)  # mono → (1, frames)

                        yield AudioRawFrame(audio=arr, sr=sr or 48000)

                    else:
                        # Handle JSON messages
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue

                        if data.get("type") == "meta":
                            sr = data.get("sr", sr or 48000)
                            channels = data.get("channels", channels)
                        elif data.get("type") == "chunk":
                            continue
                        elif data.get("type") == "end":
                            break
                        elif data.get("type") == "error":
                            print("TTS server error:", data.get("message"))
                            break
        except Exception as e:
            print("TTSWebsocketService error:", e)
            return


# --- main pipeline setup and runner ---
async def main():
    from pipecat.services.ultravox.stt import UltravoxSTTService

    # Choose STT service
    if USE_ULTRAVOX and platform.system() != "Darwin":
        print("Using Ultravox STT (not recommended on macOS).")
        stt_service = UltravoxSTTService(hf_token=HF_TOKEN)
    else:
        from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel

        stt_service = WhisperSTTServiceMLX(
            model=MLXModel.LARGE_V3_TURBO, language=Language.EN
        )
        print("Using Whisper STT fallback for macOS dev.")

    # LiveKit transport
    transport = LiveKitTransport(
        url=LIVEKIT_URL,
        token=LIVEKIT_TOKEN,
        room_name=ROOM_NAME,
        params=LiveKitParams(audio_out_enabled=True, audio_in_enabled=True),
    )

    # TTS service
    tts_service = TTSWebsocketService(
        TTS_WS_URL, chunk_size=int(os.getenv("TTS_CHUNK_SIZE", "50"))
    )

    # Build Pipecat pipeline: STT -> TTS -> LiveKit output
    pipeline = Pipeline([stt_service, tts_service, transport.output()])

    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    # optional: greet participants when they join
    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        pn = (
            participant.get("info", {}).get("userName")
            or participant.get("identity")
            or "there"
        )
        print(f"Participant joined: {pn}")
        await task.queue_frames([TextFrame(f"Hello {pn}, I'm {BOT_NAME}."), EndFrame()])

    print("Starting Pipecat runner. Join the LiveKit room:", ROOM_NAME)
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
