"""
Smoke test for a running backend: the analysis queue reaches its worker.

Creates a project, queues a folder with one image, starts processing and
opens the job's WebSocket the way the app's progress dialog does (connect,
send "ready"), then waits for the worker's first progress message. That
message is sent before any model or environment is touched, so this checks
the HTTP API, the WebSocket upgrade and the worker start, and nothing else.

    python scripts/smoke_queue_ws.py http://127.0.0.1:8765
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import httpx
import websockets
from PIL import Image


async def main(base: str) -> int:
    folder = Path(tempfile.mkdtemp(prefix="wsp-smoke-"))
    Image.new("RGB", (64, 48), (90, 110, 70)).save(folder / "IMG_0001.JPG")

    async with httpx.AsyncClient(base_url=base, timeout=30) as http:
        r = await http.post("/api/projects", json={
            "name": "smoke", "detection_model_id": "MD5A-0-0",
            "classification_model_id": None, "embedding_model_id": None,
        })
        r.raise_for_status()
        project_id = r.json()["id"]
        r = await http.post("/api/deployment-queue", json={
            "project_id": project_id, "folder_path": str(folder), "image_count": 1,
        })
        r.raise_for_status()
        r = await http.post("/api/deployment-queue/process", json={"project_id": project_id})
        r.raise_for_status()
        job_ids = r.json()["job_ids"]
        print("process:", r.json())
        if not job_ids:
            print("FAIL: no job started")
            return 1

    ws_url = base.replace("http", "ws", 1) + f"/ws/jobs/{job_ids[0]}"
    async with websockets.connect(ws_url, open_timeout=15) as ws:
        print("websocket open:", ws_url)
        await ws.send(json.dumps({"type": "ready"}))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        print("first message:", msg.get("type"), msg.get("message"))
        await ws.send(json.dumps({"type": "cancel"}))
    return 0 if msg.get("type") in ("progress", "error", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765")))
