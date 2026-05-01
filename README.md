# S2 Market Trading Test Package

This package runs the actual DREAM/S2 market exercise: it does **not** just display quotes.

The production workflow:

1. Fetches real market quote history across the broad listed-symbol universe plus cross-asset anchors.
2. Builds prior feature rows from completed market history.
3. Trains two models on prior realized moves:
   - **baseline**: conventional return/trend/volatility/drawdown priors.
   - **S2**: baseline priors plus retention-law features derived from real trailing returns.
4. Predicts next market moves for configured horizons, currently 1 and 5 trading days.
5. Restores the previous prediction state, scores prior predictions once their future return is observable, and writes a live scorecard.
6. Saves updated prediction state for the next run, then deploys the static GitHub Pages report.

No synthetic or simulated market feeds are generated. If real quote fetching fails, the run exits non-zero.

## Main workflow

Run one action:

```text
Train S2 market predictor and deploy static site
```

That single workflow trains, predicts, scores, and deploys. It does **not** push generated files to `main`, so it avoids protected-branch / pre-receive hook failures.

Required repository setting:

```text
Settings -> Pages -> Build and deployment -> Source -> GitHub Actions
```

## State handling

Because generated files are not committed to `main`, live prediction state is preserved with GitHub Actions cache:

```text
docs/data/state/prediction_state.csv
```

Each run restores the latest cache, scores predictions whose horizon has elapsed, appends new pending predictions, saves a new cache, and uploads the data as a workflow artifact.

## Outputs

Core S2 trading-test outputs:

- `docs/data/live_predictions.csv` - current pending predictions by ticker/model/horizon.
- `docs/data/prediction_state.csv` - full pending + realized prediction ledger.
- `docs/data/prediction_scorecard.csv` - realized predictions only.
- `docs/data/model_comparison.json` - baseline vs S2 held-out backtest metrics.
- `docs/data/fetch_manifest.json` - run metadata, universe size, quote rows, failures.
- `docs/index.html` - deployed static report.

Secondary diagnostics:

- `docs/data/quotes_long.csv` - real quote history used for training/scoring.
- `docs/data/signals_latest.csv` - conventional latest signal table retained only as a diagnostic.
- `docs/data/universe.csv` - resolved ticker universe.

## Local run

```bash
python -m pip install -r requirements.txt
python scripts/s2_market_predictor.py \
  --universe us-all \
  --period 2y \
  --interval 1d \
  --output-dir docs/data \
  --state-dir docs/data/state \
  --html docs/index.html
```

For a quick local smoke test with fewer symbols:

```bash
python scripts/s2_market_predictor.py --universe seed --max-tickers 200 --period 2y
```

## Model interpretation

The package measures whether the S2-enhanced model adds out-of-sample edge over the baseline.

Important fields:

- `directional_accuracy`: hit rate for non-hold predictions.
- `coverage`: share of held-out rows where the model made a BUY/SELL call instead of HOLD.
- `pnl_proxy_mean`: realized return multiplied by predicted direction. Long gains are positive; short gains are positive; wrong-way moves are negative. It is not a brokerage-grade P&L model and excludes costs/slippage.
- `MAE` / `RMSE`: prediction error on future returns.
- `delta_*`: S2 minus baseline for the same horizon.

Live predictions are not counted as wins/losses until the horizon passes. On the next run with enough real quote rows, they become realized scorecard rows and their actual moves are also available as priors for subsequent training.

## Real data only

The package refuses to fabricate output if live quote history cannot be fetched:

```text
No real market quote rows were fetched. Refusing simulated/fake outputs.
```

Unit tests use deterministic in-memory fixtures to test calculation logic; those fixtures are not market feeds and are not used by the workflow or generated outputs.
