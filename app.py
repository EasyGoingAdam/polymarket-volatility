"""FastAPI server with background price poller and SSE streaming."""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import api_client
import bollinger
import indicators
import patterns
import risk
import composite
import db
from config import MARKET_ID, POLL_INTERVAL_SECONDS, RISK_COMPUTE_INTERVAL

# Global state
yes_token_id: Optional[str] = None
market_info: Dict = {}
latest_state: Dict = {}
latest_risk: Dict = {}
poll_count: int = 0
sse_clients: List[asyncio.Queue] = []


def broadcast(payload: dict):
    dead = []
    msg = json.dumps(payload, default=str)
    for q in sse_clients:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        sse_clients.remove(q)


async def backfill_history():
    """Load historical price data into the DB."""
    if not yes_token_id:
        return
    count = await asyncio.to_thread(db.get_snapshot_count)
    if count > 50:
        print(f"[Backfill] Already have {count} snapshots, skipping")
        return

    print("[Backfill] Fetching historical prices...")
    history = await api_client.fetch_price_history(yes_token_id, interval="max", fidelity=60)
    if not history:
        print("[Backfill] No history returned")
        return

    for point in history:
        ts = point["t"]
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        await asyncio.to_thread(db.store_snapshot, {
            "timestamp": dt.isoformat(),
            "unix_ts": ts,
            "yes_price": point["p"],
            "is_backfill": 1,
        })
    print(f"[Backfill] Stored {len(history)} historical snapshots")


async def poll_loop():
    """Main polling loop - fetch price + orderbook every POLL_INTERVAL_SECONDS."""
    global latest_state, latest_risk, poll_count
    while True:
        try:
            # Fetch current data
            ob = await api_client.fetch_orderbook(yes_token_id)
            mkt = await api_client.fetch_market(MARKET_ID)

            # Determine current price
            if ob.get("best_bid") is not None and ob.get("best_ask") is not None:
                current_price = (ob["best_bid"] + ob["best_ask"]) / 2
            elif mkt.get("yes_price") is not None:
                current_price = mkt["yes_price"]
            else:
                print("[Poll] Could not determine price, skipping")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                continue

            now = datetime.now(timezone.utc)
            snapshot = {
                "timestamp": now.isoformat(),
                "unix_ts": now.timestamp(),
                "yes_price": round(current_price, 6),
                "best_bid": ob.get("best_bid"),
                "best_ask": ob.get("best_ask"),
                "spread": ob.get("spread"),
                "imbalance": ob.get("imbalance"),
                "volume_24hr": mkt.get("volume_24hr"),
                "liquidity": mkt.get("liquidity"),
            }
            await asyncio.to_thread(db.store_snapshot, snapshot)

            # Get recent snapshots for all computations
            recent = await asyncio.to_thread(db.get_recent_snapshots, 500)

            # Compute all technical indicators
            indicator_values = indicators.compute_all(recent)

            # Add pattern/regime detection
            prices = [s["yes_price"] for s in recent if s.get("yes_price")]
            adx_val = patterns.compute_adx(prices)
            hurst_val = patterns.compute_hurst(prices)
            regime = patterns.detect_regime(adx_val, hurst_val)
            indicator_values["adx"] = adx_val
            indicator_values["hurst_exponent"] = hurst_val
            indicator_values["regime"] = regime

            await asyncio.to_thread(db.store_indicator_snapshot, indicator_values)

            # Compute Bollinger bands (for chart overlay)
            bands = bollinger.compute_bollinger(recent)
            band_series = bollinger.compute_band_series(recent)

            # Generate composite signal
            comp_signal = composite.generate_signal(indicator_values, snapshot, latest_risk)
            if comp_signal:
                await asyncio.to_thread(db.store_composite_signal, comp_signal)
                print(f"[Composite] {comp_signal['signal_type']} score={comp_signal['composite_score']:.3f} "
                      f"conf={comp_signal['confidence']:.2f} regime={regime} | {comp_signal['reasoning']}")

            # Keep old Bollinger signal for backward compat
            old_signal = None
            if bands:
                old_signal = bollinger.detect_signal(current_price, bands, ob.get("imbalance", 0))
                if old_signal:
                    await asyncio.to_thread(db.store_signal, old_signal)

            # Risk metrics every N polls
            poll_count += 1
            if poll_count % RISK_COMPUTE_INTERVAL == 0:
                comp_signals = await asyncio.to_thread(db.get_recent_composite_signals, 200)
                latest_risk = risk.compute_metrics(comp_signals, recent)
                if latest_risk.get("total_signals", 0) > 0:
                    await asyncio.to_thread(db.store_risk_metrics, latest_risk)

            # Support/resistance
            sr = patterns.find_support_resistance(prices)

            latest_state = {
                "snapshot": snapshot,
                "bollinger": bands,
                "signal": comp_signal,
                "old_signal": old_signal,
                "indicators": indicator_values,
                "risk": latest_risk,
                "support_resistance": sr,
                "band_series": band_series[-100:],
                "market": {
                    "question": market_info.get("question", ""),
                    "volume": mkt.get("volume"),
                    "volume_24hr": mkt.get("volume_24hr"),
                    "liquidity": mkt.get("liquidity"),
                },
            }

            broadcast(latest_state)

        except Exception as e:
            import traceback
            print(f"[Poll] Error: {e}")
            traceback.print_exc()

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global yes_token_id, market_info

    db.init_db()
    print("[Startup] Fetching market metadata...")
    market_info = await api_client.fetch_market(MARKET_ID)
    if market_info.get("clob_token_ids"):
        yes_token_id = market_info["clob_token_ids"][0]
        print(f"[Startup] YES token: {yes_token_id}")
        print(f"[Startup] Market: {market_info.get('question')}")
        print(f"[Startup] Current YES price: {market_info.get('yes_price')}")
    else:
        print("[Startup] WARNING: Could not resolve token IDs")

    await backfill_history()
    task = asyncio.create_task(poll_loop())
    print(f"[Startup] Polling every {POLL_INTERVAL_SECONDS}s. Dashboard at http://localhost:8888")

    yield

    task.cancel()
    await api_client.close_session()


app = FastAPI(title="Polymarket Volatility Monitor", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html") as f:
        return f.read()


@app.get("/api/history")
async def api_history(limit: int = 500):
    snapshots = await asyncio.to_thread(db.get_recent_snapshots, limit)
    band_series = bollinger.compute_band_series(snapshots)
    return {"snapshots": snapshots, "band_series": band_series}


@app.get("/api/signals")
async def api_signals(limit: int = 50):
    signals = await asyncio.to_thread(db.get_recent_signals, limit)
    return {"signals": signals}


@app.get("/api/status")
async def api_status():
    count = await asyncio.to_thread(db.get_snapshot_count)
    return {
        "market_id": MARKET_ID,
        "question": market_info.get("question", ""),
        "yes_token_id": yes_token_id,
        "snapshot_count": count,
        "latest": latest_state.get("snapshot"),
        "bollinger": latest_state.get("bollinger"),
        "indicators": latest_state.get("indicators"),
        "active_signal": latest_state.get("signal"),
        "risk": latest_state.get("risk"),
        "connected_clients": len(sse_clients),
    }


@app.get("/api/indicators")
async def api_indicators():
    return {"indicators": latest_state.get("indicators", {})}


@app.get("/api/indicators/history")
async def api_indicators_history(limit: int = 200):
    data = await asyncio.to_thread(db.get_recent_indicator_snapshots, limit)
    return {"indicators": data}


@app.get("/api/risk")
async def api_risk():
    return {"risk": latest_risk}


@app.get("/api/composite")
async def api_composite(limit: int = 50):
    signals = await asyncio.to_thread(db.get_recent_composite_signals, limit)
    return {"signals": signals}


@app.get("/events")
async def sse(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    sse_clients.append(q)

    async def stream():
        try:
            # Send current state immediately
            if latest_state:
                yield f"data: {json.dumps(latest_state, default=str)}\n\n"

            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"

                if await request.is_disconnected():
                    break
        finally:
            if q in sse_clients:
                sse_clients.remove(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8888, reload=True)
