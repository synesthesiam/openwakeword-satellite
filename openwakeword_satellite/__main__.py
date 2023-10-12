#!/usr/bin/env python3
import asyncio
import argparse
import logging
import threading
from collections import deque
from queue import Queue
from typing import Tuple

from .mic import record
from .snd import play

_LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int)
    args = parser.parse_args()

    threading.Thread(target=_mic_proc, args=(args, audio_queue), daemon=True).start()

    while True:
        _detect_wake_word()


# -----------------------------------------------------------------------------


def _mic_proc(
    args: argparse.ArgumentParser,
    audio_queue: "Queue[Tuple[int, bytes]]",
) -> None:
    try:
        for ts_chunk in record(args.device):
            audio_queue.put_nowait(ts_chunk)
    except Exception:
        _LOGGER.exception("Unexpected error in _mic_proc")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
