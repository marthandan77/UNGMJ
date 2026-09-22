# UNG Forecast Machine

This repository hosts the Android-compatible cTrader WebView presentation layer. It is alert-only and never places trades.

Runtime rules:

- Reject unsigned or stale snapshots.
- Render `HOLD`/`INVALIDATED` when the data-quality field is not `valid`.
- Never place trades.
- Never embed cTrader, Telegram, email, or market-data secrets in browser code.
- GitHub Pages deployment is defined in .github/workflows/pages.yml.
