import os
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

# Load .env file
load_dotenv()

API_KEY = os.getenv("LIVEKIT_API_KEY")
API_SECRET = os.getenv("LIVEKIT_API_SECRET")
ROOM = os.getenv("LIVEKIT_ROOM", "demo-room")
IDENTITY = "vatsal"

token = (
    AccessToken(API_KEY, API_SECRET)
    .with_identity(IDENTITY)
    .with_grants(VideoGrants(room=ROOM, room_join=True))
    .to_jwt()
)

print(token)
