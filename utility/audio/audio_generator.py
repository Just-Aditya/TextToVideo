import os
import asyncio
import tempfile
from gtts import gTTS
from pydub import AudioSegment


async def generate_audio(text: str, output_filename: str):
    """
    Generate speech audio asynchronously from text.

    Args:
        text (str): Text to convert to speech.
        output_filename (str): Destination filename (supports .mp3 or .wav).

    Behavior:
        - Runs gTTS in a thread for async compatibility.
        - Converts to WAV automatically if requested.
        - Works in Colab and local Ubuntu (no websocket errors).
    """

    def _generate_sync(text: str, out_file: str):
        # Create temporary mp3
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(tmp_fd)
        try:
            print("[gTTS] Generating audio...")
            tts = gTTS(text=text, lang="en")
            tts.save(tmp_path)

            _, ext = os.path.splitext(out_file)
            ext = ext.lower()

            if ext == ".wav":
                # Convert MP3 to WAV using pydub (requires ffmpeg)
                audio = AudioSegment.from_mp3(tmp_path)
                audio.export(out_file, format="wav")
                os.remove(tmp_path)
            else:
                os.replace(tmp_path, out_file)

            print(f"[gTTS] Audio saved successfully → {out_file}")

        except Exception as e:
            print(f"[gTTS] Error: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise

    # Run blocking work in a thread so asyncio.run() calls still work
    await asyncio.to_thread(_generate_sync, text, output_filename)
