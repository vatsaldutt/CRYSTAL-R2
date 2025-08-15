# client.py
import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd
import struct

WS_URL = "wss://babc4a38a067.ngrok-free.app/ws/tts"  # change to your server/ngrok URL

TEXT_TO_SPEAK = """If you want small class sizes, personalized mentorship, and early research chances, Vanderbilt can be a great stepping stone.
But if you’re dead set on top-tier quantum AI, it won’t give you the same brand weight or research prestige as the other schools we talked about."""

async def main():
    async with websockets.connect(WS_URL) as ws:
        # Send init request
        await ws.send(json.dumps({
            "text": TEXT_TO_SPEAK,
            "chunk_size": 50
        }))

        sample_rate = None
        channels = 1
        audio_queue = []

        while True:
            msg = await ws.recv()

            if isinstance(msg, bytes):
                # This is a binary audio payload
                float_count = len(msg) // 4
                float_array = struct.unpack("<" + "f" * float_count, msg)
                audio_chunk = np.array(float_array, dtype=np.float32)

                if channels > 1:
                    audio_chunk = audio_chunk.reshape(-1, channels)

                audio_queue.append(audio_chunk)

            elif isinstance(msg, str):
                try:
                    data = json.loads(msg)
                except Exception:
                    print("Non-JSON:", msg)
                    continue

                if data["type"] == "meta":
                    sample_rate = data["sr"]
                    channels = data.get("channels", 1)
                    print(f"[meta] sr={sample_rate}, ch={channels}")

                elif data["type"] == "chunk":
                    pass  # next message will be binary data

                elif data["type"] == "end":
                    print("[server] TTS stream ended.")
                    break

                elif data["type"] == "error":
                    print("[server error]", data["message"])
                    break

        # Play everything sequentially
        if sample_rate and audio_queue:
            print(f"[playback] {len(audio_queue)} chunks, sequential")
            for chunk in audio_queue:
                sd.play(chunk, samplerate=sample_rate)
                sd.wait()  # Wait until finished before playing next

if __name__ == "__main__":
    asyncio.run(main())
