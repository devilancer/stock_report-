"""Stock Desk — 4-agent research pipeline web app."""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.agents import run_pipeline
from core.data import DataError

BASE = Path(__file__).parent
app = FastAPI(title="Stock Desk — AI Research Pipeline")


@app.get("/api/analyze/stream")
def analyze_stream(symbol: str = Query(..., min_length=1, max_length=20)):
    def gen():
        q: "queue.Queue[dict]" = queue.Queue()

        def worker():
            try:
                run_pipeline(symbol.strip(), q.put)
            except DataError as e:
                q.put({"type": "error", "message": str(e)})
            except Exception as e:  # noqa: BLE001
                q.put({"type": "error", "message": f"분석 중 오류가 발생했습니다: {type(e).__name__}: {e}"})
            finally:
                q.put({"type": "end"})

        threading.Thread(target=worker, daemon=True).start()
        while True:
            ev = q.get()
            if ev.get("type") == "end":
                break
            name = ev.get("type", "log")
            yield f"event: {name}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/health")
def health():
    return JSONResponse({"ok": True})


app.mount("/", StaticFiles(directory=str(BASE / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8787)
