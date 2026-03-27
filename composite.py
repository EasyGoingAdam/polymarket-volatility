"""Composite signal engine — multi-indicator weighted scoring with regime adaptation."""
from __future__ import annotations

import statistics
from typing import Optional, List, Dict
from datetime import datetime, timezone

from config import (
    COMPOSITE_SIGNAL_THRESHOLD, REGIME_TRENDING_ADX,
    WEIGHT_MEAN_REVERSION, WEIGHT_MOMENTUM, WEIGHT_VOLATILITY,
    WEIGHT_ORDER_FLOW, WEIGHT_VOLUME, KELLY_FRACTION,
)


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Component scorers (each returns -1 to +1)
# ---------------------------------------------------------------------------

def _mean_reversion_score(ind: Dict) -> float:
    """Bollinger z-score (inverted) + RSI + CCI + Williams %R → consensus."""
    scores = []

    z = ind.get("bollinger_z")
    if z is not None:
        scores.append(_clamp(-z / 2.0))  # inverted: negative z = buy = positive score

    rsi = ind.get("rsi_14")
    if rsi is not None:
        # RSI 30 → +1 (oversold=buy), RSI 70 → -1 (overbought=sell), 50 → 0
        scores.append(_clamp((50 - rsi) / 20.0))

    cci = ind.get("cci_20")
    if cci is not None:
        # CCI -100 → +1, CCI +100 → -1
        scores.append(_clamp(-cci / 100.0))

    wr = ind.get("williams_r")
    if wr is not None:
        # Williams %R: -80 to -100 → oversold → +1; 0 to -20 → overbought → -1
        scores.append(_clamp((-50 - wr) / 30.0))

    return round(statistics.mean(scores), 4) if scores else 0.0


def _momentum_score(ind: Dict) -> float:
    """MACD histogram + ROC + Stochastic crossover → consensus."""
    scores = []

    hist = ind.get("macd_histogram")
    if hist is not None:
        # Normalize: typical histogram range for prediction markets is tiny
        scores.append(_clamp(hist / 0.01))

    roc = ind.get("roc_12")
    if roc is not None:
        scores.append(_clamp(roc / 5.0))  # 5% ROC = full score

    k = ind.get("stoch_k")
    d = ind.get("stoch_d")
    if k is not None and d is not None:
        # %K above %D = bullish
        diff = (k - d) / 20.0
        scores.append(_clamp(diff))

    return round(statistics.mean(scores), 4) if scores else 0.0


def _volatility_score(ind: Dict) -> float:
    """ATR expansion/contraction + vol-of-vol + Keltner position → regime context.
    Positive = volatility contracting (favorable for entry), negative = expanding."""
    scores = []

    # Vol-of-vol: high vol-of-vol = uncertainty = negative score
    vov = ind.get("vol_of_vol")
    if vov is not None:
        scores.append(_clamp(-vov * 10))  # scale factor

    # Keltner position: price within channels = calm, outside = volatile
    ku = ind.get("keltner_upper")
    kl = ind.get("keltner_lower")
    km = ind.get("keltner_mid")
    bz = ind.get("bollinger_z")
    if ku and kl and km and bz is not None:
        # Inside Keltner = positive (low vol), outside = negative
        rng = ku - kl
        if rng > 1e-6:
            pos = abs(bz)
            scores.append(_clamp(1.0 - pos))

    return round(statistics.mean(scores), 4) if scores else 0.0


def _orderflow_score(snapshot: Dict) -> float:
    """Order book imbalance + spread → directional signal."""
    scores = []

    imb = snapshot.get("imbalance")
    if imb is not None:
        # Direct mapping: positive imbalance = buy pressure = positive score
        scores.append(_clamp(imb * 2.0))

    spread = snapshot.get("spread")
    if spread is not None:
        # Tighter spread = more conviction = boost magnitude
        # Don't add direction, just confidence multiplier handled elsewhere
        pass

    return round(statistics.mean(scores), 4) if scores else 0.0


def _volume_score(ind: Dict) -> float:
    """OBV direction + volume ROC → conviction."""
    scores = []

    obv = ind.get("obv")
    if obv is not None:
        # Positive OBV = accumulation = bullish
        scores.append(_clamp(obv / 50000.0))  # normalize to reasonable range

    vroc = ind.get("volume_roc")
    if vroc is not None:
        # Rising volume = more conviction in current direction
        scores.append(_clamp(vroc / 20.0))

    return round(statistics.mean(scores), 4) if scores else 0.0


# ---------------------------------------------------------------------------
# Composite Signal Generator
# ---------------------------------------------------------------------------

def generate_signal(indicators: Dict, snapshot: Dict,
                    risk_metrics: Optional[Dict] = None) -> Optional[Dict]:
    """Generate composite buy/sell signal from all indicators.
    Returns signal dict for DB storage, or None if below threshold."""
    now = datetime.now(timezone.utc)

    # Compute component scores
    mr_score = _mean_reversion_score(indicators)
    mom_score = _momentum_score(indicators)
    vol_score = _volatility_score(indicators)
    of_score = _orderflow_score(snapshot)
    v_score = _volume_score(indicators)

    # Regime-adaptive weights
    regime = indicators.get("regime", "unknown")
    adx = indicators.get("adx")

    w_mr = WEIGHT_MEAN_REVERSION
    w_mom = WEIGHT_MOMENTUM
    w_vol = WEIGHT_VOLATILITY
    w_of = WEIGHT_ORDER_FLOW
    w_v = WEIGHT_VOLUME

    if regime == "trending":
        # Boost momentum, reduce mean reversion
        w_mom += 0.10
        w_mr -= 0.10
    elif regime == "mean_reverting":
        # Boost mean reversion, reduce momentum
        w_mr += 0.10
        w_mom -= 0.10

    # Weighted composite score
    composite = (
        mr_score * w_mr +
        mom_score * w_mom +
        vol_score * w_vol +
        of_score * w_of +
        v_score * w_v
    )
    composite = round(_clamp(composite), 4)

    # Confidence: agreement among components
    component_scores = [mr_score, mom_score, vol_score, of_score, v_score]
    non_zero = [s for s in component_scores if abs(s) > 0.01]
    if len(non_zero) >= 2:
        # Low std dev = high agreement = high confidence
        std = statistics.stdev(non_zero)
        confidence = round(max(0, 1.0 - std), 4)
    else:
        confidence = 0.2

    # Check threshold
    if abs(composite) < COMPOSITE_SIGNAL_THRESHOLD:
        return None

    signal_type = "BUY" if composite > 0 else "SELL"

    # Kelly / risk info
    kelly = 0.0
    sharpe = None
    sortino = None
    max_dd = None
    var_95 = None
    if risk_metrics:
        kelly = risk_metrics.get("kelly_fraction", 0.0) or 0.0
        sharpe = risk_metrics.get("sharpe_ratio")
        sortino = risk_metrics.get("sortino_ratio")
        max_dd = risk_metrics.get("max_drawdown")
        var_95 = risk_metrics.get("var_95")

    suggested_pct = round(kelly * confidence * 100, 2)  # % of bankroll

    # Build reasoning
    parts = []
    if abs(mr_score) > 0.2:
        parts.append(f"MeanRev {'bullish' if mr_score > 0 else 'bearish'}({mr_score:+.2f})")
    if abs(mom_score) > 0.2:
        parts.append(f"Momentum {'bullish' if mom_score > 0 else 'bearish'}({mom_score:+.2f})")
    if abs(of_score) > 0.2:
        parts.append(f"OrderFlow {'buy' if of_score > 0 else 'sell'}({of_score:+.2f})")
    if regime != "unknown":
        parts.append(f"Regime={regime}")
    reasoning = " | ".join(parts) if parts else "Weak consensus"

    return {
        "timestamp": now.isoformat(),
        "unix_ts": now.timestamp(),
        "signal_type": signal_type,
        "composite_score": composite,
        "confidence": confidence,
        "mean_reversion_score": mr_score,
        "momentum_score": mom_score,
        "volatility_score": vol_score,
        "orderflow_score": of_score,
        "volume_score": v_score,
        "kelly_fraction": kelly,
        "suggested_position_pct": suggested_pct,
        "current_sharpe": sharpe,
        "current_sortino": sortino,
        "max_drawdown": max_dd,
        "var_95": var_95,
        "regime": regime,
        "adx": adx,
        "reasoning": reasoning,
    }
