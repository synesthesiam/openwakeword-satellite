import argparse
import asyncio
import sys
import time
from typing import Final, Optional, Iterable, Tuple

import sounddevice as sd
from speexdsp_ns import NoiseSuppression

_RATE: Final = 16000
_CHANNELS: Final = 1


def record(
    device: Optional[int],
    samples_per_chunk: int = 1280,
    suppress_noise: bool = True,
) -> Iterable[Tuple[int, bytes]]:
    """Yield mic samples with a timestamp."""
    if suppress_noise:
        speex_ns: Optional[NoiseSuppression] = NoiseSuppression.create(
            samples_per_chunk, _RATE
        )
    else:
        speex_ns = None

    with sd.RawInputStream(
        device=device,
        samplerate=_RATE,
        channels=_CHANNELS,
        blocksize=samples_per_chunk,
        dtype="int16",
    ) as stream:
        while True:
            chunk, _overflowed = stream.read(samples_per_chunk)
            if suppress_noise:
                chunk = speex_ns.process(bytes(chunk))
            else:
                chunk = bytes(chunk)

            yield time.monotonic_ns(), chunk


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int)
    args = parser.parse_args()

    print("Ready", file=sys.stderr)
    for _timestamp, chunk in record(args.device):
        sys.stdout.buffer.write(chunk)
