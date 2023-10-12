import argparse
import subprocess
import sys
import wave
from typing import Optional

import sounddevice as sd


def play(media: str, device: Optional[int], samples_per_chunk: int = 1024) -> None:
    proc = subprocess.Popen(
        ["ffmpeg", "-i", media, "-f", "wav", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    with wave.open(proc.stdout, "rb") as wav_file:
        assert wav_file.getsampwidth() == 2
        with sd.RawOutputStream(
            device=device,
            samplerate=wav_file.getframerate(),
            channels=wav_file.getnchannels(),
            dtype="int16",
        ) as device:
            chunk = wav_file.readframes(samples_per_chunk)
            while chunk:
                device.write(chunk)
                chunk = wav_file.readframes(samples_per_chunk)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("media")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()

    print("Ready", file=sys.stderr)
    play(args.media, args.device)
