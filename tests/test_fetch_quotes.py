import datetime as dt
import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_quotes.py"
spec = importlib.util.spec_from_file_location("fetch_quotes", SCRIPT)
fetch_quotes = importlib.util.module_from_spec(spec)
import sys
sys.modules["fetch_quotes"] = fetch_quotes
spec.loader.exec_module(fetch_quotes)


class FetchQuotesTests(unittest.TestCase):
    def test_seed_universe_is_not_tiny(self):
        tickers = fetch_quotes.dedupe_preserve_order(fetch_quotes.FALLBACK_TICKERS)
        self.assertGreater(len(tickers), 200)
        self.assertIn("SPY", tickers)
        self.assertIn("AAPL", tickers)
        self.assertIn("^VIX", tickers)

    def test_parse_nasdaq_pipe_file_excludes_test_issue_and_footer(self):
        text = (
            "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
            "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
            "ZZTEST|Some Test Issue|Q|Y|N|100|N|N\n"
            "ABCW|ABC Warrants|Q|N|N|100|N|N\n"
            "File Creation Time: 0501202600|||||||\n"
        )
        symbols = fetch_quotes.parse_nasdaq_pipe_file(text, "test", include_special=False)
        self.assertEqual(symbols, ["AAPL"])

    def test_compute_latest_signals_uses_real_rows_without_fake_outputs(self):
        # Deterministic fixture rows. Tests the calculation path only; package outputs
        # are created only from live downloads in scripts/fetch_quotes.py.
        rows = []
        start = dt.date(2025, 1, 1)
        for i in range(90):
            price = 100.0 + i * 0.5
            rows.append({
                "date": (start + dt.timedelta(days=i)).isoformat(),
                "ticker": "AAA",
                "open": price - 0.25,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "adj_close": price,
                "volume": 1000000 + i,
                "source": "fixture",
            })
            price2 = 120.0 - i * 0.3
            rows.append({
                "date": (start + dt.timedelta(days=i)).isoformat(),
                "ticker": "BBB",
                "open": price2 + 0.25,
                "high": price2 + 0.5,
                "low": price2 - 0.5,
                "close": price2,
                "adj_close": price2,
                "volume": 900000 + i,
                "source": "fixture",
            })
        signals = fetch_quotes.compute_latest_signals(rows, min_history_days=60)
        self.assertEqual({r["ticker"] for r in signals}, {"AAA", "BBB"})
        top = signals[0]
        self.assertEqual(top["ticker"], "AAA")
        self.assertGreater(top["market_signal_score"], signals[1]["market_signal_score"])
        summary = fetch_quotes.build_summary(signals, {
            "universe_source": "test", "requested_tickers": 2, "successful_tickers": 2, "failed_tickers": 0
        })
        self.assertEqual(summary["signals_count"], 2)
        self.assertIn("breadth", summary)

    def test_html_report_includes_charts_and_explainer(self):
        rows = [
            {"ticker": "SPY", "signal_label": "strong", "market_signal_score": 80.0, "last_close": 500.0, "ret_5d": 0.01, "ret_20d": 0.04, "ret_63d": 0.02, "above_sma50": 1, "above_sma200": 1, "realized_vol_20d": 0.12, "rsi14": 65.0},
            {"ticker": "AAA", "signal_label": "weak", "market_signal_score": 30.0, "last_close": 25.0, "ret_5d": -0.02, "ret_20d": -0.06, "ret_63d": -0.08, "above_sma50": 0, "above_sma200": 0, "realized_vol_20d": 0.35, "rsi14": 35.0},
        ]
        summary = {
            "generated_at_utc": "2026-05-01T00:00:00+00:00",
            "latest_market_date": "2026-05-01",
            "universe_source": "test",
            "requested_tickers": 2,
            "successful_tickers": 2,
            "signals_count": 2,
            "breadth": {"pct_above_sma20": 0.5, "pct_above_sma50": 0.5, "pct_above_sma200": 0.5, "pct_positive_20d": 0.5, "pct_near_20d_high": 0.5, "pct_near_20d_low": 0.0, "median_ret_20d": -0.01},
            "anchors": [rows[0]],
            "top_signals": rows,
            "bottom_signals": list(reversed(rows)),
        }
        out = ROOT / "docs" / "test_report.html"
        fetch_quotes.write_html_report(out, summary)
        html = out.read_text()
        self.assertIn("Market breadth", html)
        self.assertIn("Cross-asset anchor scores", html)
        self.assertIn("How to read this", html)
        self.assertIn("Top market signals", html)


if __name__ == "__main__":
    unittest.main()
