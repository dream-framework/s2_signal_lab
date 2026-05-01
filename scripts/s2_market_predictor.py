#!/usr/bin/env python3
"""
Run the live S2 market trading test.

The workflow is intentionally not a quote display pipeline:
1) fetch real market history,
2) train a baseline model on completed prior market moves,
3) train an S2-enhanced model on the same priors plus retention features,
4) predict next moves,
5) score prior predictions once their future move is observable,
6) preserve prediction state so realized gains/losses become future priors.

No synthetic or simulated market feeds are generated. If real quote fetching fails,
the run exits non-zero.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# Reuse the hardened live-fetching and quote-signal helpers from v4.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import fetch_quotes as fq  # noqa: E402

BASELINE_FEATURES = [
    "ret_1d", "ret_5d", "ret_20d", "ret_63d", "ret_126d",
    "distance_sma20", "distance_sma50", "distance_sma200",
    "realized_vol_20d", "rsi14", "drawdown_63d", "drawdown_252d",
    "distance_52w_high", "distance_52w_low",
]

S2_FEATURES = [
    "s2_retention_5", "s2_retention_20", "s2_retention_63",
    "s2_beta_proxy", "s2_lambda_proxy", "s2_fit_r2",
]

ALL_FEATURES = BASELINE_FEATURES + S2_FEATURES
PREDICTION_FIELDS = [
    "prediction_id", "created_at_utc", "model", "horizon", "ticker",
    "asof_date", "target_date", "asof_close", "target_close",
    "predicted_return", "predicted_direction", "trade_signal", "confidence",
    "actual_return", "actual_direction", "hit", "pnl_proxy", "abs_error",
    "status", "source_run_id",
]


def log(message: str) -> None:
    print(message, flush=True)


def to_float(value: Any) -> float | None:
    return fq.to_float(value)


def mean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else None


def stddev(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return None
    return statistics.stdev(vals)


def safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0 or not math.isfinite(float(b)):
        return None
    out = float(a) / float(b)
    return out if math.isfinite(out) else None


def pct_change(closes: list[float], idx: int, lookback: int) -> float | None:
    prev_idx = idx - lookback
    if prev_idx < 0 or idx >= len(closes):
        return None
    prev = closes[prev_idx]
    cur = closes[idx]
    if prev <= 0 or cur <= 0:
        return None
    return cur / prev - 1.0


def future_return(closes: list[float], idx: int, horizon: int) -> float | None:
    fut_idx = idx + horizon
    if fut_idx >= len(closes):
        return None
    cur = closes[idx]
    fut = closes[fut_idx]
    if cur <= 0 or fut <= 0:
        return None
    return fut / cur - 1.0


def max_drawdown(values: list[float]) -> float | None:
    return fq.max_drawdown(values)


def simple_rsi(values: list[float], period: int = 14) -> float | None:
    return fq.simple_rsi(values, period)


def rounded(value: Any, digits: int = 8) -> Any:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return round(float(value), digits)
    return value


def rolling_mean_series(values: list[float], window: int) -> list[float]:
    if window <= 0 or len(values) < window:
        return []
    out: list[float] = []
    running = sum(values[:window])
    out.append(running / window)
    for i in range(window, len(values)):
        running += values[i] - values[i - window]
        out.append(running / window)
    return out


def retention_ratio(returns: list[float], window: int) -> float | None:
    vals = [r for r in returns if math.isfinite(r)]
    if len(vals) < max(window + 5, 20):
        return None
    raw_std = stddev(vals)
    if raw_std is None or raw_std <= 0:
        return None
    # Fast block-mean attenuation proxy. This keeps the production workflow
    # tractable for thousands of symbols while still measuring real-prior
    # retention across scale W.
    blocks = []
    start = max(0, len(vals) - max(window * 8, window + 5))
    tail = vals[start:]
    for i in range(0, len(tail) - window + 1, window):
        block = tail[i:i + window]
        if len(block) == window:
            blocks.append(statistics.fmean(block))
    if len(blocks) < 2:
        # Fallback to the latest block mean magnitude when not enough blocks exist.
        latest = abs(statistics.fmean(vals[-window:]))
        ratio = latest / raw_std
    else:
        block_std = stddev(blocks)
        if block_std is None:
            return None
        ratio = block_std / raw_std
    if not math.isfinite(ratio):
        return None
    return max(0.0, min(1.5, ratio))


def linear_fit_r2(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None, None
    xbar = statistics.fmean(xs)
    ybar = statistics.fmean(ys)
    ssx = sum((x - xbar) ** 2 for x in xs)
    if ssx <= 0:
        return None, None
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / ssx
    intercept = ybar - slope * xbar
    sst = sum((y - ybar) ** 2 for y in ys)
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - sse / sst if sst > 0 else None
    return slope, r2


def s2_retention_features(trailing_returns: list[float]) -> dict[str, float | None]:
    """Proxy S2 features from how return volatility attenuates under smoothing scale.

    These are not a simulated feed. They are deterministic features derived from
    real trailing returns available as of the prediction date.
    """
    vals = [r for r in trailing_returns if math.isfinite(r)]
    ratios = {w: retention_ratio(vals, w) for w in (5, 20, 63)}
    xs: list[float] = []
    ys: list[float] = []
    for w in (2, 5, 10, 20, 40, 63):
        ratio = retention_ratio(vals, w)
        if ratio is None or ratio <= 0 or ratio >= 0.999:
            continue
        # For stretched exponential S(W) ~= exp(-(W/lambda)^beta),
        # log(-log(S)) = beta * log(W) - beta * log(lambda).
        xs.append(math.log(float(w)))
        ys.append(math.log(-math.log(ratio)))
    beta, r2 = linear_fit_r2(xs, ys)
    lambda_proxy = None
    if beta is not None and beta != 0 and xs and ys:
        xbar = statistics.fmean(xs)
        ybar = statistics.fmean(ys)
        intercept = ybar - beta * xbar
        try:
            lambda_proxy = math.exp(-intercept / beta)
        except Exception:
            lambda_proxy = None
    return {
        "s2_retention_5": ratios.get(5),
        "s2_retention_20": ratios.get(20),
        "s2_retention_63": ratios.get(63),
        "s2_beta_proxy": beta,
        "s2_lambda_proxy": lambda_proxy,
        "s2_fit_r2": r2,
    }


def feature_row(ticker: str, rows: list[dict[str, Any]], idx: int, min_history_days: int) -> dict[str, Any] | None:
    if idx < min_history_days or idx >= len(rows):
        return None
    closes = [to_float(r.get("adj_close")) or to_float(r.get("close")) for r in rows]
    if any(v is None or v <= 0 for v in closes[: idx + 1]):
        return None
    close_vals = [float(v) for v in closes]
    cur = close_vals[idx]
    prior = close_vals[: idx + 1]
    returns = [(prior[i] / prior[i - 1]) - 1.0 for i in range(1, len(prior)) if prior[i - 1] > 0]
    trailing_returns = returns[-252:]
    sma20 = mean(prior[-20:]) if len(prior) >= 20 else None
    sma50 = mean(prior[-50:]) if len(prior) >= 50 else None
    sma200 = mean(prior[-200:]) if len(prior) >= 200 else None
    high_252 = max(prior[-252:]) if len(prior) >= 252 else max(prior)
    low_252 = min(prior[-252:]) if len(prior) >= 252 else min(prior)
    row = {
        "ticker": ticker,
        "date": rows[idx].get("date"),
        "close": cur,
        "ret_1d": pct_change(close_vals, idx, 1),
        "ret_5d": pct_change(close_vals, idx, 5),
        "ret_20d": pct_change(close_vals, idx, 20),
        "ret_63d": pct_change(close_vals, idx, 63),
        "ret_126d": pct_change(close_vals, idx, 126),
        "distance_sma20": safe_div(cur - sma20, sma20) if sma20 is not None else None,
        "distance_sma50": safe_div(cur - sma50, sma50) if sma50 is not None else None,
        "distance_sma200": safe_div(cur - sma200, sma200) if sma200 is not None else None,
        "realized_vol_20d": (stddev(returns[-20:]) or 0.0) * math.sqrt(252.0) if len(returns) >= 20 else None,
        "rsi14": simple_rsi(prior, 14),
        "drawdown_63d": max_drawdown(prior[-63:]),
        "drawdown_252d": max_drawdown(prior[-252:]),
        "distance_52w_high": safe_div(cur - high_252, high_252),
        "distance_52w_low": safe_div(cur - low_252, low_252),
    }
    row.update(s2_retention_features(trailing_returns))
    # Replace missing with neutral zero after model features are calculated.
    for key in ALL_FEATURES:
        val = row.get(key)
        if val is None or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            row[key] = 0.0
        else:
            row[key] = float(val)
    return row


def group_quotes(quotes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in quotes:
        ticker = row.get("ticker")
        close = to_float(row.get("adj_close")) or to_float(row.get("close"))
        if ticker and close is not None and close > 0:
            grouped[str(ticker)].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: r.get("date") or "")
    return dict(grouped)


def build_examples(grouped: dict[str, list[dict[str, Any]]], horizon: int, min_history_days: int, train_lookback_days: int, train_stride: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    latest_features: list[dict[str, Any]] = []
    for ticker, rows in grouped.items():
        if len(rows) < min_history_days + horizon + 2:
            continue
        n = len(rows)
        start_idx = max(min_history_days, n - train_lookback_days - horizon)
        for idx in range(start_idx, n - horizon, max(1, train_stride)):
            feat = feature_row(ticker, rows, idx, min_history_days)
            target = future_return([float(to_float(r.get("adj_close")) or to_float(r.get("close"))) for r in rows], idx, horizon)
            if feat is None or target is None:
                continue
            feat["target_return"] = target
            feat["target_direction"] = 1 if target > 0 else -1 if target < 0 else 0
            feat["target_date"] = rows[idx + horizon].get("date")
            examples.append(feat)
        live_idx = n - 1
        live_feat = feature_row(ticker, rows, live_idx, min_history_days)
        if live_feat is not None:
            latest_features.append(live_feat)
    examples.sort(key=lambda r: (r.get("date") or "", r.get("ticker") or ""))
    latest_features.sort(key=lambda r: r.get("ticker") or "")
    return examples, latest_features


@dataclass
class Standardizer:
    means: list[float]
    scales: list[float]

    def transform_row(self, row: list[float]) -> list[float]:
        return [(x - m) / s for x, m, s in zip(row, self.means, self.scales)]


def feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> list[list[float]]:
    return [[float(row.get(name, 0.0) or 0.0) for name in feature_names] for row in rows]


def fit_standardizer(matrix: list[list[float]]) -> Standardizer:
    if not matrix:
        return Standardizer([], [])
    cols = len(matrix[0])
    means: list[float] = []
    scales: list[float] = []
    for j in range(cols):
        vals = [row[j] for row in matrix]
        mu = statistics.fmean(vals) if vals else 0.0
        sd = statistics.stdev(vals) if len(vals) > 1 else 1.0
        if not math.isfinite(sd) or sd <= 1e-12:
            sd = 1.0
        means.append(mu)
        scales.append(sd)
    return Standardizer(means, scales)


def apply_standardizer(matrix: list[list[float]], standardizer: Standardizer) -> list[list[float]]:
    return [standardizer.transform_row(row) for row in matrix]


def solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    # Augmented matrix with partial pivoting.
    mat = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < 1e-12:
            mat[pivot][col] = 1e-12
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
        pivot_val = mat[col][col]
        for j in range(col, n + 1):
            mat[col][j] /= pivot_val
        for r in range(n):
            if r == col:
                continue
            factor = mat[r][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                mat[r][j] -= factor * mat[col][j]
    return [mat[i][n] for i in range(n)]


@dataclass
class RidgeModel:
    feature_names: list[str]
    standardizer: Standardizer
    coefficients: list[float]
    residual_std: float

    def predict_one(self, row: dict[str, Any]) -> float:
        raw = [float(row.get(name, 0.0) or 0.0) for name in self.feature_names]
        x = [1.0] + self.standardizer.transform_row(raw)
        return sum(c * v for c, v in zip(self.coefficients, x))


def fit_ridge(rows: list[dict[str, Any]], feature_names: list[str], ridge_lambda: float = 1.0) -> RidgeModel:
    if not rows:
        return RidgeModel(feature_names, Standardizer([0.0] * len(feature_names), [1.0] * len(feature_names)), [0.0] * (len(feature_names) + 1), 0.02)
    x_raw = feature_matrix(rows, feature_names)
    standardizer = fit_standardizer(x_raw)
    x = [[1.0] + row for row in apply_standardizer(x_raw, standardizer)]
    y = [float(row.get("target_return", 0.0) or 0.0) for row in rows]
    p = len(x[0])
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]
    for xi, yi in zip(x, y):
        for i in range(p):
            xty[i] += xi[i] * yi
            for j in range(p):
                xtx[i][j] += xi[i] * xi[j]
    for i in range(1, p):  # do not penalize intercept
        xtx[i][i] += ridge_lambda
    try:
        coeffs = solve_linear_system(xtx, xty)
    except Exception:
        coeffs = [statistics.fmean(y) if y else 0.0] + [0.0] * (p - 1)
    residuals = []
    for xi, yi in zip(x, y):
        pred = sum(c * v for c, v in zip(coeffs, xi))
        residuals.append(yi - pred)
    residual_std = stddev(residuals) or 0.02
    return RidgeModel(feature_names, standardizer, coeffs, residual_std)


def prediction_direction(predicted_return: float, threshold: float) -> int:
    if predicted_return > threshold:
        return 1
    if predicted_return < -threshold:
        return -1
    return 0


def direction_label(direction: int) -> str:
    return "BUY" if direction > 0 else "SELL" if direction < 0 else "HOLD"


def confidence_score(predicted_return: float, residual_std: float) -> float:
    denom = max(abs(residual_std), 1e-6)
    z = abs(predicted_return) / denom
    return round(max(0.0, min(100.0, 50.0 + 18.0 * z)), 2)


def evaluate_predictions(rows: list[dict[str, Any]], model: RidgeModel, threshold: float) -> dict[str, Any]:
    if not rows:
        return {"test_rows": 0}
    preds = [model.predict_one(row) for row in rows]
    actuals = [float(row.get("target_return", 0.0) or 0.0) for row in rows]
    errors = [p - a for p, a in zip(preds, actuals)]
    directions = [prediction_direction(p, threshold) for p in preds]
    actual_dirs = [1 if a > 0 else -1 if a < 0 else 0 for a in actuals]
    directional_rows = [(d, ad) for d, ad in zip(directions, actual_dirs) if d != 0 and ad != 0]
    hits = [1 for d, ad in directional_rows if d == ad]
    pnl = [a * d for a, d in zip(actuals, directions) if d != 0]
    buy_actuals = [a for a, d in zip(actuals, directions) if d > 0]
    sell_actuals = [-a for a, d in zip(actuals, directions) if d < 0]
    return {
        "test_rows": len(rows),
        "mae": rounded(mean([abs(e) for e in errors]) or 0.0),
        "rmse": rounded(math.sqrt(mean([e * e for e in errors]) or 0.0)),
        "directional_accuracy": rounded(len(hits) / len(directional_rows) if directional_rows else None),
        "coverage": rounded(sum(1 for d in directions if d != 0) / len(directions) if directions else 0.0),
        "pnl_proxy_mean": rounded(mean(pnl) or 0.0),
        "buy_count": sum(1 for d in directions if d > 0),
        "sell_count": sum(1 for d in directions if d < 0),
        "hold_count": sum(1 for d in directions if d == 0),
        "avg_realized_when_buy": rounded(mean(buy_actuals) or 0.0),
        "avg_realized_when_sell": rounded(mean(sell_actuals) or 0.0),
    }


def train_and_predict_for_horizon(
    examples: list[dict[str, Any]],
    latest_features: list[dict[str, Any]],
    horizon: int,
    asof_date: str,
    threshold: float,
    backtest_fraction: float,
    max_training_rows: int,
    run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not examples or not latest_features:
        return [], {"horizon": horizon, "error": "not enough examples or latest features"}

    # Time split: train on older priors, test on the latest completed outcomes.
    unique_dates = sorted({str(r.get("date")) for r in examples if r.get("date")})
    split_at = max(1, int(len(unique_dates) * (1.0 - backtest_fraction)))
    cutoff_date = unique_dates[min(split_at, len(unique_dates) - 1)]
    train_rows = [r for r in examples if str(r.get("date")) < cutoff_date]
    test_rows = [r for r in examples if str(r.get("date")) >= cutoff_date]
    if not train_rows or not test_rows:
        split_n = max(1, int(len(examples) * (1.0 - backtest_fraction)))
        train_rows, test_rows = examples[:split_n], examples[split_n:]
    if max_training_rows > 0 and len(train_rows) > max_training_rows:
        # Deterministic downsampling across the prior set. This keeps broad market coverage
        # without turning the workflow into a multi-hour compute job.
        step = len(train_rows) / max_training_rows
        train_rows = [train_rows[int(i * step)] for i in range(max_training_rows)]

    models = {
        "baseline": fit_ridge(train_rows, BASELINE_FEATURES),
        "s2": fit_ridge(train_rows, ALL_FEATURES),
    }
    metrics: dict[str, Any] = {
        "horizon": horizon,
        "asof_date": asof_date,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "cutoff_date": cutoff_date,
        "threshold": threshold,
        "models": {},
    }
    live_rows: list[dict[str, Any]] = []
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    for model_name, model in models.items():
        eval_metrics = evaluate_predictions(test_rows, model, threshold)
        metrics["models"][model_name] = eval_metrics
        for row in latest_features:
            pred = model.predict_one(row)
            direction = prediction_direction(pred, threshold)
            live_rows.append({
                "prediction_id": f"{model_name}|{row['ticker']}|{row['date']}|h{horizon}",
                "created_at_utc": now,
                "model": model_name,
                "horizon": horizon,
                "ticker": row["ticker"],
                "asof_date": row["date"],
                "target_date": "",
                "asof_close": rounded(row.get("close"), 6),
                "target_close": "",
                "predicted_return": rounded(pred, 8),
                "predicted_direction": direction,
                "trade_signal": direction_label(direction),
                "confidence": confidence_score(pred, model.residual_std),
                "actual_return": "",
                "actual_direction": "",
                "hit": "",
                "pnl_proxy": "",
                "abs_error": "",
                "status": "pending",
                "source_run_id": run_id,
            })
    base = metrics["models"].get("baseline", {})
    s2 = metrics["models"].get("s2", {})
    metrics["s2_vs_baseline"] = {
        "delta_directional_accuracy": rounded((s2.get("directional_accuracy") or 0.0) - (base.get("directional_accuracy") or 0.0)),
        "delta_pnl_proxy_mean": rounded((s2.get("pnl_proxy_mean") or 0.0) - (base.get("pnl_proxy_mean") or 0.0)),
        "delta_mae": rounded((s2.get("mae") or 0.0) - (base.get("mae") or 0.0)),
        "s2_better_directional_accuracy": bool((s2.get("directional_accuracy") or 0.0) > (base.get("directional_accuracy") or 0.0)),
        "s2_better_pnl_proxy": bool((s2.get("pnl_proxy_mean") or 0.0) > (base.get("pnl_proxy_mean") or 0.0)),
    }
    return live_rows, metrics


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    fq.write_csv(path, rows, fieldnames)


def save_json(path: Path, obj: Any) -> None:
    fq.save_json(path, obj)


def score_existing_predictions(existing: list[dict[str, Any]], grouped: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], int]:
    newly_scored = 0
    close_by_ticker_date: dict[str, dict[str, tuple[int, float]]] = {}
    rows_by_ticker: dict[str, list[tuple[str, float]]] = {}
    for ticker, rows in grouped.items():
        series: list[tuple[str, float]] = []
        for i, row in enumerate(rows):
            close = to_float(row.get("adj_close")) or to_float(row.get("close"))
            if close is not None and close > 0:
                series.append((str(row.get("date")), float(close)))
        rows_by_ticker[ticker] = series
        close_by_ticker_date[ticker] = {date: (idx, close) for idx, (date, close) in enumerate(series)}

    for pred in existing:
        if pred.get("status") == "realized":
            continue
        ticker = str(pred.get("ticker") or "")
        asof_date = str(pred.get("asof_date") or "")
        try:
            horizon = int(float(pred.get("horizon") or 1))
        except Exception:
            horizon = 1
        series = rows_by_ticker.get(ticker) or []
        by_date = close_by_ticker_date.get(ticker) or {}
        if asof_date not in by_date:
            continue
        asof_idx, asof_close = by_date[asof_date]
        target_idx = asof_idx + horizon
        if target_idx >= len(series):
            continue
        target_date, target_close = series[target_idx]
        actual = target_close / asof_close - 1.0
        actual_direction = 1 if actual > 0 else -1 if actual < 0 else 0
        try:
            pred_direction = int(float(pred.get("predicted_direction") or 0))
        except Exception:
            pred_direction = 0
        pred_ret = to_float(pred.get("predicted_return")) or 0.0
        pred["target_date"] = target_date
        pred["target_close"] = rounded(target_close, 6)
        pred["actual_return"] = rounded(actual, 8)
        pred["actual_direction"] = actual_direction
        pred["hit"] = int(pred_direction == actual_direction) if pred_direction != 0 and actual_direction != 0 else ""
        pred["pnl_proxy"] = rounded(actual * pred_direction, 8) if pred_direction != 0 else 0.0
        pred["abs_error"] = rounded(abs(pred_ret - actual), 8)
        pred["status"] = "realized"
        newly_scored += 1
    return existing, newly_scored


def merge_predictions(existing: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {str(r.get("prediction_id")): dict(r) for r in existing if r.get("prediction_id")}
    for row in new_rows:
        pid = str(row.get("prediction_id"))
        if pid not in by_id:
            by_id[pid] = dict(row)
    merged = list(by_id.values())
    merged.sort(key=lambda r: (str(r.get("asof_date") or ""), str(r.get("horizon") or ""), str(r.get("model") or ""), str(r.get("ticker") or "")))
    return merged


def summarize_realized(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    realized = [p for p in predictions if p.get("status") == "realized"]
    summary: dict[str, Any] = {"realized_predictions": len(realized), "by_model_horizon": {}}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in realized:
        groups[(str(row.get("model")), str(row.get("horizon")))].append(row)
    for (model, horizon), rows in sorted(groups.items()):
        hits = [int(float(r.get("hit"))) for r in rows if str(r.get("hit")) not in ("", "None")]
        pnl = [to_float(r.get("pnl_proxy")) for r in rows if to_float(r.get("pnl_proxy")) is not None]
        abs_err = [to_float(r.get("abs_error")) for r in rows if to_float(r.get("abs_error")) is not None]
        summary["by_model_horizon"][f"{model}_h{horizon}"] = {
            "count": len(rows),
            "directional_accuracy": rounded(sum(hits) / len(hits) if hits else None),
            "pnl_proxy_mean": rounded(mean(pnl) or 0.0),
            "mae": rounded(mean(abs_err) or 0.0),
        }
    return summary


def fmt_pct(value: Any) -> str:
    if isinstance(value, str) and value == "":
        return ""
    v = to_float(value)
    if v is not None and math.isfinite(v):
        return f"{100.0 * v:.2f}%"
    return ""


def escape_html(value: Any) -> str:
    return fq.escape_html(value)


def metric_cell(value: Any, pct: bool = False) -> str:
    if pct:
        return escape_html(fmt_pct(value))
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return escape_html(round(float(value), 6))
    return escape_html(value)


def write_html_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_runs = report.get("model_runs") or []
    live = report.get("live_predictions") or []
    realized_summary = report.get("realized_summary") or {}
    manifest = report.get("manifest") or {}
    latest_date = report.get("latest_market_date")

    def comparison_table() -> str:
        rows_html = []
        for run in model_runs:
            horizon = run.get("horizon")
            models = run.get("models") or {}
            for model in ("baseline", "s2"):
                m = models.get(model) or {}
                rows_html.append(
                    "<tr>"
                    f"<td>{escape_html(horizon)}</td>"
                    f"<td>{escape_html(model)}</td>"
                    f"<td>{escape_html(run.get('train_rows'))}</td>"
                    f"<td>{escape_html(m.get('test_rows'))}</td>"
                    f"<td>{metric_cell(m.get('directional_accuracy'), pct=True)}</td>"
                    f"<td>{metric_cell(m.get('coverage'), pct=True)}</td>"
                    f"<td>{metric_cell(m.get('pnl_proxy_mean'), pct=True)}</td>"
                    f"<td>{metric_cell(m.get('mae'), pct=True)}</td>"
                    f"<td>{metric_cell(m.get('rmse'), pct=True)}</td>"
                    "</tr>"
                )
        return """
<section><h2>Backtest: S2 model vs baseline</h2><div class='table-wrap'><table>
<thead><tr><th>horizon</th><th>model</th><th>prior train rows</th><th>held-out rows</th><th>direction hit</th><th>coverage</th><th>PnL proxy</th><th>MAE</th><th>RMSE</th></tr></thead>
<tbody>""" + "".join(rows_html) + "</tbody></table></div></section>"

    def delta_cards() -> str:
        cards = []
        for run in model_runs:
            delta = run.get("s2_vs_baseline") or {}
            cards.append((f"h{run.get('horizon')} Δ hit", fmt_pct(delta.get("delta_directional_accuracy"))))
            cards.append((f"h{run.get('horizon')} Δ PnL", fmt_pct(delta.get("delta_pnl_proxy_mean"))))
            cards.append((f"h{run.get('horizon')} Δ MAE", fmt_pct(delta.get("delta_mae"))))
        return "".join(f"<div class='card'><span>{escape_html(k)}</span><b>{escape_html(v)}</b></div>" for k, v in cards)

    def prediction_table(rows: list[dict[str, Any]], title: str) -> str:
        cols = ["ticker", "horizon", "model", "trade_signal", "predicted_return", "confidence", "asof_date", "asof_close"]
        body = []
        for row in rows:
            cells = []
            for col in cols:
                value = row.get(col)
                if col == "predicted_return":
                    value = fmt_pct(value)
                elif col == "confidence" and value != "":
                    value = f"{float(value):.1f}" if isinstance(value, (int, float)) or str(value).replace('.', '', 1).isdigit() else value
                cells.append(f"<td>{escape_html(value)}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        header = "".join(f"<th>{escape_html(c)}</th>" for c in cols)
        return f"<section><h2>{escape_html(title)}</h2><div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div></section>"

    s2_live = [r for r in live if r.get("model") == "s2"]
    top_buys = sorted([r for r in s2_live if r.get("trade_signal") == "BUY"], key=lambda r: to_float(r.get("predicted_return")) or 0.0, reverse=True)[:40]
    top_sells = sorted([r for r in s2_live if r.get("trade_signal") == "SELL"], key=lambda r: to_float(r.get("predicted_return")) or 0.0)[:40]
    high_conf = sorted(s2_live, key=lambda r: (to_float(r.get("confidence")) or 0.0), reverse=True)[:40]

    cards = [
        ("Latest market date", latest_date),
        ("Requested tickers", manifest.get("requested_tickers")),
        ("Successful tickers", manifest.get("successful_tickers")),
        ("Quote rows", manifest.get("quote_rows")),
        ("Live predictions", len(live)),
        ("Prior predictions scored", report.get("newly_scored_predictions")),
        ("Total realized scores", realized_summary.get("realized_predictions")),
        ("Universe", manifest.get("universe_source")),
    ]
    card_html = "".join(f"<div class='card'><span>{escape_html(k)}</span><b>{escape_html(v)}</b></div>" for k, v in cards)

    realized_rows = []
    for key, val in (realized_summary.get("by_model_horizon") or {}).items():
        realized_rows.append(
            "<tr>"
            f"<td>{escape_html(key)}</td>"
            f"<td>{escape_html(val.get('count'))}</td>"
            f"<td>{metric_cell(val.get('directional_accuracy'), pct=True)}</td>"
            f"<td>{metric_cell(val.get('pnl_proxy_mean'), pct=True)}</td>"
            f"<td>{metric_cell(val.get('mae'), pct=True)}</td>"
            "</tr>"
        )
    realized_table = """
<section><h2>Live scorecard from prior predictions</h2><div class='table-wrap'><table>
<thead><tr><th>model/horizon</th><th>realized rows</th><th>direction hit</th><th>PnL proxy</th><th>MAE</th></tr></thead>
<tbody>""" + "".join(realized_rows) + "</tbody></table></div></section>"

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>S2 Market Predictor</title>
<style>
:root {{ color-scheme: light dark; --bg:#0f172a; --panel:#111827; --panel2:#0b1220; --text:#e5e7eb; --muted:#94a3b8; --line:#334155; --accent:#60a5fa; }}
body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--text); }}
main {{ max-width: 1500px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 4px; font-size:30px; }}
p {{ color:var(--muted); margin:0 0 16px; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr)); gap:12px; margin:18px 0; }}
.card, .explain {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:14px; }}
.card span {{ display:block; color:var(--muted); font-size:12px; }}
.card b {{ font-size:18px; }}
section {{ margin:24px 0; }}
h2 {{ font-size:18px; margin:0 0 10px; }}
.table-wrap {{ width:100%; overflow-x:auto; border:1px solid var(--line); border-radius:14px; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); }}
th, td {{ padding:8px 10px; border-bottom:1px solid var(--line); font-size:13px; text-align:right; white-space:nowrap; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3), th:nth-child(4), td:nth-child(4) {{ text-align:left; }}
th {{ color:var(--muted); font-weight:600; }}
.explain {{ color:var(--muted); line-height:1.5; }}
.explain b {{ color:var(--text); }}
.footer {{ margin-top:30px; color:var(--muted); font-size:12px; }}
</style></head><body><main>
<h1>S2 Market Trading Test</h1>
<p>Real market data only. Generated {escape_html(report.get('generated_at_utc'))}. This page trains on prior completed market moves, predicts next moves, and scores prior predictions when the future return becomes observable.</p>
<div class='grid'>{card_html}</div>
<div class='grid'>{delta_cards()}</div>
<section><h2>How to read this</h2><div class='explain'>
<b>Baseline</b> uses conventional prior features: returns, trend distance, volatility, RSI, and drawdown. <b>S2</b> uses the same priors plus retention-law features derived from how realized return variance attenuates across smoothing windows. The live predictions are not scored until their horizon has passed. On later runs, those realized gains/losses are marked in the scorecard and are also available as new training labels because they have become part of the historical quote record.
</div></section>
{comparison_table()}
{realized_table}
{prediction_table(top_buys, 'S2 live next-move BUY predictions')}
{prediction_table(top_sells, 'S2 live next-move SELL predictions')}
{prediction_table(high_conf, 'S2 highest-confidence live predictions')}
<div class='footer'>Outputs: data/live_predictions.csv, data/prediction_state.csv, data/prediction_scorecard.csv, data/model_comparison.json, data/quotes_long.csv. No simulated feeds.</div>
</main></body></html>
"""
    path.write_text(html, encoding="utf-8")


def write_scorecard(path: Path, predictions: list[dict[str, Any]]) -> None:
    realized = [p for p in predictions if p.get("status") == "realized"]
    write_csv(path, realized, PREDICTION_FIELDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/test S2 market predictor on real quote priors and deploy outputs.")
    parser.add_argument("--universe", choices=["us-all", "seed", "anchors", "custom"], default=os.getenv("MARKET_UNIVERSE", "us-all"))
    parser.add_argument("--tickers-file", default=os.getenv("MARKET_TICKERS_FILE", "data/tickers.csv"))
    parser.add_argument("--output-dir", default=os.getenv("MARKET_OUTPUT_DIR", "docs/data"))
    parser.add_argument("--state-dir", default=os.getenv("MARKET_STATE_DIR", "docs/data/state"))
    parser.add_argument("--html", default=os.getenv("MARKET_HTML", "docs/index.html"))
    parser.add_argument("--period", default=os.getenv("MARKET_PERIOD", "2y"))
    parser.add_argument("--interval", default=os.getenv("MARKET_INTERVAL", "1d"))
    parser.add_argument("--sources", default=os.getenv("MARKET_SOURCES", "yahoo,stooq"))
    parser.add_argument("--workers", type=int, default=int(os.getenv("MARKET_WORKERS", "12")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("MARKET_RETRIES", "2")))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("MARKET_TIMEOUT", "30")))
    parser.add_argument("--max-tickers", type=int, default=int(os.getenv("MARKET_MAX_TICKERS", "0")))
    parser.add_argument("--min-history-days", type=int, default=int(os.getenv("MARKET_MIN_HISTORY_DAYS", "120")))
    parser.add_argument("--train-lookback-days", type=int, default=int(os.getenv("MARKET_TRAIN_LOOKBACK_DAYS", "160")))
    parser.add_argument("--train-stride", type=int, default=int(os.getenv("MARKET_TRAIN_STRIDE", "5")), help="use every Nth prior row per ticker for model training examples")
    parser.add_argument("--max-training-rows", type=int, default=int(os.getenv("MARKET_MAX_TRAINING_ROWS", "180000")))
    parser.add_argument("--horizons", default=os.getenv("MARKET_HORIZONS", "1,5"), help="comma-separated trading-day prediction horizons")
    parser.add_argument("--prediction-threshold", type=float, default=float(os.getenv("MARKET_PREDICTION_THRESHOLD", "0.0015")))
    parser.add_argument("--backtest-fraction", type=float, default=float(os.getenv("MARKET_BACKTEST_FRACTION", "0.25")))
    parser.add_argument("--progress-every", type=int, default=int(os.getenv("MARKET_PROGRESS_EVERY", "100")))
    parser.add_argument("--include-special", action="store_true")
    parser.add_argument("--strict-universe", action="store_true")
    args = parser.parse_args(argv)
    args.sources = [s.strip().lower() for s in str(args.sources).split(",") if s.strip()]
    horizons = [int(h.strip()) for h in str(args.horizons).split(",") if h.strip()]
    run_id = os.getenv("GITHUB_RUN_ID") or dt.datetime.now(dt.timezone.utc).strftime("local-%Y%m%d%H%M%S")

    out_dir = Path(args.output_dir)
    state_dir = Path(args.state_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    tickers, universe_source, source_note = fq.resolve_universe(args)
    if not tickers:
        log("[ERROR] Universe resolved to zero tickers.")
        return 2
    if source_note:
        log(f"[WARN] {source_note}")
    write_csv(out_dir / "universe.csv", [{"ticker": t} for t in tickers], ["ticker"])

    all_rows, failures = fq.fetch_all_quotes(tickers, args)
    if not all_rows:
        log("[ERROR] No real market quote rows were fetched. Refusing simulated/fake outputs.")
        save_json(out_dir / "fetch_manifest.json", {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "universe_source": universe_source,
            "requested_tickers": len(tickers),
            "successful_tickers": 0,
            "failed_tickers": len(failures),
            "failures": failures[:500],
        })
        return 3

    all_rows.sort(key=lambda r: (r.get("ticker") or "", r.get("date") or ""))
    quote_fields = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
    write_csv(out_dir / "quotes_long.csv", all_rows, quote_fields)
    grouped = group_quotes(all_rows)
    latest_market_date = max((row.get("date") or "" for row in all_rows), default="")

    # Keep the old latest signal table as a secondary diagnostic, not the main product.
    signals = fq.compute_latest_signals(all_rows, min(60, args.min_history_days))
    signal_fields = [
        "ticker", "last_date", "last_close", "rows", "source", "signal_label", "market_signal_score",
        "ret_1d", "ret_5d", "ret_20d", "ret_63d", "ret_126d", "ret_252d",
        "sma20", "sma50", "sma200", "above_sma20", "above_sma50", "above_sma200",
        "distance_sma50", "distance_sma200", "realized_vol_20d", "rsi14", "drawdown_252d",
        "near_20d_high", "near_20d_low", "distance_52w_high", "distance_52w_low", "avg_volume_20d",
        "rank_ret_5d", "rank_ret_20d", "rank_ret_63d", "rank_ret_126d", "rank_low_vol_20d", "rank_drawdown_252d",
    ]
    if signals:
        write_csv(out_dir / "signals_latest.csv", signals, signal_fields)

    existing_state_path = state_dir / "prediction_state.csv"
    existing_predictions = read_csv(existing_state_path)
    scored_predictions, newly_scored = score_existing_predictions(existing_predictions, grouped)

    live_predictions: list[dict[str, Any]] = []
    model_runs: list[dict[str, Any]] = []
    for horizon in horizons:
        examples, latest_features = build_examples(grouped, horizon, args.min_history_days, args.train_lookback_days, args.train_stride)
        log(f"[INFO] Horizon h{horizon}: examples={len(examples)} latest_features={len(latest_features)}")
        horizon_predictions, metrics = train_and_predict_for_horizon(
            examples=examples,
            latest_features=latest_features,
            horizon=horizon,
            asof_date=latest_market_date,
            threshold=args.prediction_threshold,
            backtest_fraction=args.backtest_fraction,
            max_training_rows=args.max_training_rows,
            run_id=str(run_id),
        )
        live_predictions.extend(horizon_predictions)
        model_runs.append(metrics)

    if not live_predictions:
        log("[ERROR] Real quotes were fetched, but no live predictions could be trained. Check min history / period.")
        return 4

    merged_state = merge_predictions(scored_predictions, live_predictions)
    write_csv(state_dir / "prediction_state.csv", merged_state, PREDICTION_FIELDS)
    write_csv(out_dir / "prediction_state.csv", merged_state, PREDICTION_FIELDS)
    write_csv(out_dir / "live_predictions.csv", live_predictions, PREDICTION_FIELDS)
    write_scorecard(out_dir / "prediction_scorecard.csv", merged_state)

    successful = len({row["ticker"] for row in all_rows})
    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "run_id": run_id,
        "universe_source": universe_source,
        "universe_note": source_note,
        "requested_tickers": len(tickers),
        "successful_tickers": successful,
        "failed_tickers": len(failures),
        "quote_rows": len(all_rows),
        "latest_market_date": latest_market_date,
        "period": args.period,
        "interval": args.interval,
        "sources": args.sources,
        "horizons": horizons,
        "newly_scored_predictions": newly_scored,
        "failures": failures[:1000],
    }
    save_json(out_dir / "fetch_manifest.json", manifest)
    comparison = {
        "generated_at_utc": manifest["generated_at_utc"],
        "latest_market_date": latest_market_date,
        "model_runs": model_runs,
        "realized_summary": summarize_realized(merged_state),
        "newly_scored_predictions": newly_scored,
    }
    save_json(out_dir / "model_comparison.json", comparison)
    report = dict(comparison)
    report["manifest"] = manifest
    report["live_predictions"] = live_predictions[:]
    report["generated_at_utc"] = manifest["generated_at_utc"]
    if args.html:
        write_html_report(Path(args.html), report)

    log(f"[INFO] Completed S2 market test. requested={len(tickers)} successful={successful} live_predictions={len(live_predictions)} newly_scored={newly_scored}")
    log(f"[INFO] Wrote {out_dir / 'live_predictions.csv'} and {out_dir / 'model_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
