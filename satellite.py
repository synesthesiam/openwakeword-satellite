#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import audioop
import logging
import shlex
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from queue import Queue
from typing import Deque, Optional, Iterable
from pathlib import Path

import aiohttp
import numpy as np
from openwakeword.model import Model

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)


@dataclass
class State:
    """Client state."""

    args: argparse.Namespace
    oww_model: Model
    running: bool = True
    recording: bool = False
    loop: asyncio.AbstractEventLoop = field(default_factory=asyncio.get_running_loop)
    wake_queue: "Queue[Optional[bytes]]" = field(default_factory=Queue)
    pipeline_queue: "asyncio.Queue[Optional[bytes]]" = field(
        default_factory=asyncio.Queue
    )


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rate",
        required=True,
        type=int,
        help="Rate of input audio (hertz)",
    )
    parser.add_argument(
        "--width",
        required=True,
        type=int,
        help="Width of input audio samples (bytes)",
    )
    parser.add_argument(
        "--channels",
        required=True,
        type=int,
        help="Number of input audio channels",
    )
    parser.add_argument(
        "--samples-per-chunk",
        type=int,
        default=1280,
        help="Number of samples to read at a time from stdin",
    )
    parser.add_argument(
        "--chunks-to-buffer",
        type=int,
        default=5,
    )
    #
    parser.add_argument("--awake-sound", default=_DIR / "sounds" / "awake.wav")
    parser.add_argument("--done-sound", default=_DIR / "sounds" / "done.wav")
    #
    parser.add_argument("--wake-threshold", type=float, default=0.5)
    parser.add_argument("--wake-trigger-level", type=int, default=5)
    #
    parser.add_argument("--token", required=True, help="HA auth token")
    parser.add_argument(
        "--pipeline", help="Name of HA pipeline to use (default: preferred)"
    )
    parser.add_argument(
        "--server", default="localhost:8123", help="host:port of HA server"
    )
    parser.add_argument("--server-protocol", default="http")
    #
    parser.add_argument("--play-program", default="mpg123 -q {url}")
    #
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print DEBUG messages to console",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    # Load wake word model(s)
    oww_model = Model(
        wakeword_model_paths=["models/nabu_casa.onnx"],
        vad_threshold=0.5,
        custom_verifier_models={"nabu_casa": "models/nabu_casa.pkl"},
        custom_verifier_threshold=0.3,
    )

    # Start reading raw audio from stdin
    state = State(args=args, oww_model=oww_model)

    wake_thread = threading.Thread(target=detect_wakeword, args=(state,), daemon=True)
    wake_thread.start()

    audio_thread = threading.Thread(target=read_audio, args=(state,), daemon=True)
    audio_thread.start()

    audio_buffer: "Deque[bytes]" = deque(maxlen=state.args.chunks_to_buffer)

    try:
        while state.running:
            audio_bytes = await state.pipeline_queue.get()
            if not audio_bytes:
                break

            if not state.recording:
                audio_buffer.append(audio_bytes)
                continue

            try:
                await run_pipeline(state, audio_buffer)
            except KeyboardInterrupt:
                break
            except Exception:
                _LOGGER.exception("Unexpected error while running pipeline")

            audio_buffer.clear()
            state.recording = False

            # Clear audio queue
            while not state.pipeline_queue.empty():
                await state.pipeline_queue.get()

    except KeyboardInterrupt:
        pass
    finally:
        state.recording = False
        state.running = False
        wake_thread.join()
        audio_thread.join()


async def run_pipeline(state: State, audio_buffer: Deque[bytes]) -> None:
    "Runs a single iteration of a pipeline."
    args = state.args
    url = f"ws://{args.server}/api/websocket"
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as websocket:
            _LOGGER.debug("Authenticating")
            msg = await websocket.receive_json()
            assert msg["type"] == "auth_required", msg

            await websocket.send_json(
                {
                    "type": "auth",
                    "access_token": args.token,
                }
            )

            msg = await websocket.receive_json()
            _LOGGER.debug(msg)
            assert msg["type"] == "auth_ok", msg
            _LOGGER.info("Authenticated")

            message_id = 1
            pipeline_id: Optional[str] = None
            if args.pipeline:
                # Get list of available pipelines and resolve name
                await websocket.send_json(
                    {
                        "type": "assist_pipeline/pipeline/list",
                        "id": message_id,
                    }
                )
                msg = await websocket.receive_json()
                _LOGGER.debug(msg)
                message_id += 1

                pipelines = msg["result"]["pipelines"]
                for pipeline in pipelines:
                    if pipeline["name"] == args.pipeline:
                        pipeline_id = pipeline["id"]
                        break

                if not pipeline_id:
                    raise ValueError(
                        f"No pipeline named {args.pipeline} in {pipelines}"
                    )

            # Run pipeline
            _LOGGER.debug("Starting pipeline")
            pipeline_args = {
                "type": "assist_pipeline/run",
                "id": message_id,
                "start_stage": "stt",
                "end_stage": "tts",
                "input": {
                    "sample_rate": 16000,
                },
            }
            if pipeline_id:
                pipeline_args["pipeline"] = pipeline_id
            await websocket.send_json(pipeline_args)
            message_id += 1

            msg = await websocket.receive_json()
            _LOGGER.debug(msg)
            assert msg["success"], "Pipeline failed to run"

            # Get handler id.
            # This is a single byte prefix that needs to be in every binary payload.
            msg = await websocket.receive_json()
            _LOGGER.debug(msg)
            handler_id = bytes(
                [msg["event"]["data"]["runner_data"]["stt_binary_handler_id"]]
            )

            # Audio loop for single pipeline run
            receive_event_task = asyncio.create_task(websocket.receive_json())
            while state.running:
                if audio_buffer:
                    # Use buffer first
                    audio_chunk: Optional[bytes] = audio_buffer.popleft()
                else:
                    audio_chunk = await state.pipeline_queue.get()

                if not audio_chunk:
                    break

                # Prefix binary message with handler id
                send_audio_task = asyncio.create_task(
                    websocket.send_bytes(handler_id + audio_chunk)
                )
                pending = {send_audio_task, receive_event_task}
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if receive_event_task in done:
                    event = receive_event_task.result()
                    _LOGGER.debug(event)
                    event_type = event["event"]["type"]
                    if event_type == "run-end":
                        _LOGGER.debug("Pipeline finished")
                        break

                    if event_type == "error":
                        raise RuntimeError(event["event"]["data"]["message"])

                    if event_type == "tts-end":
                        # URL of text to speech audio response (relative to server)
                        tts_url = event["event"]["data"]["tts_output"]["url"]
                        _LOGGER.debug("TTS URL: %s", tts_url)
                        play_command_str = args.play_program.format(
                            url=f"{args.server_protocol}://{args.server}{tts_url}"
                        )
                        play_command = shlex.split(play_command_str)
                        proc = await asyncio.create_subprocess_exec(
                            play_command[0], *play_command[1:]
                        )
                        await proc.communicate()

                    receive_event_task = asyncio.create_task(websocket.receive_json())

                if send_audio_task not in done:
                    await send_audio_task


def detect_wakeword(state: State) -> None:
    """Detects wake word in a loop."""
    args = state.args
    oww_model = state.oww_model
    try:
        while state.running:
            audio_bytes = state.wake_queue.get()
            if not audio_bytes:
                break

            if state.recording:
                # Silence
                audio_bytes = bytes(len(audio_bytes))

            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            prediction = oww_model.predict(audio_int16)

            if state.recording:
                continue

            for model_key in prediction:
                scores = list(oww_model.prediction_buffer[model_key])
                if (len(scores) >= args.wake_trigger_level) and all(
                    s > args.wake_threshold for s in scores[-args.wake_trigger_level :]
                ):
                    _LOGGER.info("Wake up: %s", model_key)
                    state.recording = True
    except Exception:
        _LOGGER.exception("Unexpected error detecting wake word")
        state.running = False


def read_audio(state: State) -> None:
    """Reads chunks of raw audio from standard input."""
    try:
        args = state.args
        bytes_per_chunk = args.samples_per_chunk * args.width * args.channels
        rate = args.rate
        width = args.width
        channels = args.channels
        ratecv_state = None

        _LOGGER.debug("Reading audio from stdin")

        while state.running:
            chunk = sys.stdin.buffer.read(bytes_per_chunk)
            if (not chunk) or (not state.running):
                # Signal other threads to stop
                state.wake_queue.put_nowait(None)
                state.loop.call_soon_threadsafe(state.pipeline_queue.put_nowait, None)
                break

            # Convert to 16Khz, 16-bit, mono
            if channels != 1:
                chunk = audioop.tomono(chunk, width, 1.0, 1.0)

            if width != 2:
                chunk = audioop.lin2lin(chunk, width, 2)

            if rate != 16000:
                chunk, ratecv_state = audioop.ratecv(
                    chunk,
                    2,
                    1,
                    rate,
                    16000,
                    ratecv_state,
                )

            state.wake_queue.put_nowait(chunk)
            state.loop.call_soon_threadsafe(state.pipeline_queue.put_nowait, chunk)
    except Exception:
        _LOGGER.exception("Unexpected error reading audio")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
