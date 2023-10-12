import argparse
import asyncio
import logging
import sys
import time
from typing import Any, AsyncIterable, Dict, Optional, Tuple

import aiohttp

_LOGGER = logging.getLogger(__name__)


async def stream(
    host: str,
    token: str,
    audio: "asyncio.Queue[Tuple[int, bytes]]",
    port: int = 8123,
    pipeline_name: Optional[str] = None,
) -> Tuple[int, str, Dict[str, Any]]:
    url = f"ws://{host}:{port}/api/websocket"
    message_id = 1

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as websocket:
            await _authenticate(websocket, token)

            pipeline_id: Optional[str] = None
            if pipeline_name:
                message_id, pipeline_id = await _get_pipeline_id(
                    websocket, message_id, pipeline_name
                )

            message_id, handler_id = await _start_pipeline(
                websocket, message_id, pipeline_id
            )

            async for timestamp, event_type, event_data in _audio_to_events(
                websocket, handler_id, audio
            ):
                yield timestamp, event_type, event_data


async def _authenticate(websocket, token: str):
    msg = await websocket.receive_json()
    _LOGGER.debug(msg)
    assert msg["type"] == "auth_required", msg
    await websocket.send_json(
        {
            "type": "auth",
            "access_token": token,
        }
    )

    msg = await websocket.receive_json()
    _LOGGER.debug(msg)
    assert msg["type"] == "auth_ok", msg


async def _get_pipeline_id(
    websocket, message_id: int, pipeline_name: str
) -> Tuple[int, Optional[str]]:
    pipeline_id: Optional[str] = None
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
        _LOGGER.warning("No pipeline named %s in %s", pipeline_name, pipelines)

    return message_id, pipeline_id


async def _start_pipeline(
    websocket, message_id: int, pipeline_id: Optional[str]
) -> Tuple[int, int]:
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
    handler_id = msg["event"]["data"]["runner_data"]["stt_binary_handler_id"]

    return message_id, handler_id


async def _audio_to_events(
    websocket,
    handler_id: int,
    audio: "asyncio.Queue[Tuple[int, bytes]]",
) -> Tuple[int, str, Dict[str, Any]]:
    prefix_bytes = bytes([handler_id])

    audio_task = asyncio.create_task(audio.get())
    event_task = asyncio.create_task(websocket.receive_json())
    pending = {audio_task, event_task}

    while True:
        done, pending = await asyncio.wait(
            pending,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if audio_task in done:
            # Forward to websocket
            _timestamp, audio_chunk = audio_task.result()
            pending.add(
                asyncio.create_task(websocket.send_bytes(prefix_bytes + audio_chunk))
            )

            # Next audio chunk
            audio_task = asyncio.create_task(audio.get())
            pending.add(audio_task)

        if event_task in done:
            event = event_task.result()
            if event.get("type") != "event":
                continue

            _LOGGER.debug(event)
            event_type = event["event"]["type"]
            event_data = event["event"]["data"]
            yield time.monotonic_ns(), event_type, event_data

            if event_type == "run-end":
                _LOGGER.debug("Pipeline finished")
                break

            if event_type == "error":
                _LOGGER.error(event_data["message"])
                break

            # Next event
            event_task = asyncio.create_task(websocket.receive_json())
            pending.add(event_task)

    for task in pending:
        task.cancel()


# -----------------------------------------------------------------------------


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("token")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--device", type=int)
    parser.add_argument("--pipeline")
    args = parser.parse_args()

    import threading
    from .mic import record

    loop = asyncio.get_running_loop()
    audio: "asyncio.Queue[Tuple[int, bytes]]" = asyncio.Queue()

    def audio_proc():
        for timestamp, audio_chunk in record(args.device):
            loop.call_soon_threadsafe(audio.put_nowait, (timestamp, audio_chunk))

    threading.Thread(target=audio_proc, daemon=True).start()

    while True:
        print("Ready", file=sys.stderr)
        async for timestamp, event_type, event_data in stream(
            args.host,
            args.token,
            audio,
            port=args.port,
            pipeline_name=args.pipeline,
        ):
            print(timestamp, event_type, event_data)


if __name__ == "__main__":
    asyncio.run(_main())
