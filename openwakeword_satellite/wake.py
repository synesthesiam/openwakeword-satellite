import argparse
import sys
from collections import Counter
from typing import Final, Iterable, Tuple

import numpy as np
from openwakeword.model import Model

SAMPLES_PER_CHUNK: Final = 1280
BYTES_PER_CHUNK: Final = SAMPLES_PER_CHUNK * 2  # 16-bit


def detect(
    model: Model,
    audio: Iterable[Tuple[int, bytes]],
    threshold: float = 0.5,
    trigger_level: int = 4,
    refractory_level: int = 30,
) -> Tuple[int, str]:
    activations: "Counter[str]" = Counter()

    for timestamp, chunk in audio:
        if not chunk:
            break

        assert len(chunk) == BYTES_PER_CHUNK

        model.predict(np.frombuffer(chunk, dtype=np.int16))
        for model_key, model_score in model.prediction_buffer.items():
            if model_score[-1] >= threshold:
                # Activated
                activations[model_key] += 1
            else:
                # Decay back to 0
                activations[model_key] = max(0, activations[model_key])

            if activations[model_key] >= trigger_level:
                yield timestamp, model_key
                activations[model_key] = -refractory_level


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()

    from .mic import record

    model = Model(wakeword_models=[args.model])
    audio = record(args.device, samples_per_chunk=SAMPLES_PER_CHUNK)

    print("Ready", file=sys.stderr)
    for timestamp, model_key in detect(model, audio):
        print(timestamp, model_key)
