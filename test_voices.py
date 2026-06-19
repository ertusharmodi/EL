from elevenlabs import ElevenLabs
import os

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

voices = client.voices.get_all()

for v in voices.voices:
    print(f"{v.name} -> {v.voice_id}")