import datetime as dt
import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "s2_market_predictor.py"
spec = importlib.util.spec_from_file_location("s2_market_predictor", SCRIPT)
s2 = importlib.util.module_from_spec(spec)
sys.modules["s2_market_predictor"] = s2
spec.loader.exec_module(s2)


def fixture_quotes():
    rows = []
    start = dt.date(2025, 1, 1)
    for i in range(180):
        date = (start + dt.timedelta(days=i)).isoformat()
        # AAA has persistent positive priors and mostly positive next returns.
        price_a = 100.0 + i * 0.35 + (i % 7) * 0.03
        # BBB has persistent negative priors.
        price_b = 160.0 - i * 0.22 + (i % 5) * 0.02
        for ticker, price in (("AAA", price_a), ("BBB", price_b)):
            rows.append({
                "date": date,
                "ticker": ticker,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "adj_close": price,
                "volume": 1000000 + i,
                "source": "fixture",
            })
    return rows


class S2MarketPredictorTests(unittest.TestCase):
    def test_build_examples_and_live_predictions(self):
        grouped = s2.group_quotes(fixture_quotes())
        examples, latest = s2.build_examples(grouped, horizon=1, min_history_days=60, train_lookback_days=120)
        self.assertGreater(len(examples), 40)
        self.assertEqual({r["ticker"] for r in latest}, {"AAA", "BBB"})
        preds, metrics = s2.train_and_predict_for_horizon(
            examples=examples,
            latest_features=latest,
            horizon=1,
            asof_date="2025-06-29",
            threshold=0.0005,
            backtest_fraction=0.25,
            max_training_rows=10000,
            run_id="test",
        )
        self.assertEqual(len(preds), 4)  # two tickers x baseline/S2
        self.assertIn("s2_vs_baseline", metrics)
        self.assertIn("baseline", metrics["models"])
        self.assertIn("s2", metrics["models"])

    def test_score_existing_predictions_when_future_exists(self):
        grouped = s2.group_quotes(fixture_quotes())
        existing = [{
            "prediction_id": "s2|AAA|2025-04-15|h5",
            "created_at_utc": "2025-04-15T22:00:00+00:00",
            "model": "s2",
            "horizon": "5",
            "ticker": "AAA",
            "asof_date": "2025-04-15",
            "target_date": "",
            "asof_close": "",
            "target_close": "",
            "predicted_return": "0.01",
            "predicted_direction": "1",
            "trade_signal": "BUY",
            "confidence": "75",
            "actual_return": "",
            "actual_direction": "",
            "hit": "",
            "pnl_proxy": "",
            "abs_error": "",
            "status": "pending",
            "source_run_id": "test",
        }]
        scored, newly = s2.score_existing_predictions(existing, grouped)
        self.assertEqual(newly, 1)
        self.assertEqual(scored[0]["status"], "realized")
        self.assertNotEqual(scored[0]["actual_return"], "")
        self.assertNotEqual(scored[0]["pnl_proxy"], "")


if __name__ == "__main__":
    unittest.main()
