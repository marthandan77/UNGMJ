# UNG Forecast Machine

A research-first, human-readable multi-horizon forecasting application for UNG.

The project is built against a locked quantitative specification. It forecasts five horizons—60 minutes, 4 hours, 1 trading day, 2 trading days, and 7 trading days—using calibrated probabilistic models and auditable decision logic.

## Current deployment target

- Source control and CI: GitHub
- Application: Streamlit
- Runtime market data: Twelve Data by default; Schwab remains supported
- Research fallback: yfinance, only when explicitly configured
- Persistence: local Parquet and append-only JSONL during the prototype stage
- Operating mode: research and shadow only

PostgreSQL and Oracle Cloud are deferred until the Streamlit build runs successfully end to end.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
streamlit run app.py
```

For the default configuration, enter the Twelve Data API key under `[twelvedata]` in `.streamlit/secrets.toml`. The real secrets file is excluded by `.gitignore` and must never be committed.

## Streamlit Community Cloud

1. Deploy the GitHub repository and select `app.py`.
2. Open App settings > Secrets.
3. Copy the `[twelvedata]` structure from `.streamlit/secrets.toml.example`.
4. Insert the API key.
5. Start the app and use **Load and validate Twelve Data data**.

To use Schwab instead, change `data.provider` in `config.yaml` to `schwab` and provide the `[schwab]` secrets section.

The app remains in research-only status until each horizon passes its own purged walk-forward statistical validation.
