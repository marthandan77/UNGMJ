# UNG Harvest Radar — cTrader WebView Plugin

This is the Android-compatible cTrader WebView plugin. It uses the official
Plugin SDK client-host bridge and performs the harvest calculation in the
browser using cTrader data only.

## Live cTrader bridge

`ctrader-sdk-adapter.js` performs the documented confirm/register/confirm
handshake, resolves `UNG.US`, requests 30 daily trendbars, subscribes to live
quotes, and emits these internal events:

- `ung-radar-sdk-connected`
- `ung-radar-symbol-metadata`
- `ung-radar-daily-bars`
- `ung-radar-quote`
- `ung-radar-sdk-error`

`local-engine.js` uses the daily bars for the harvest zones and the quote stream
only for the current executable mid-price. It never places trades.

## Registration

Deploy this directory at an HTTPS URL, then create a cTrader web-based plugin
from the Algo/Plugins area. Select mobile placement (Symbol Overview embedded
block or bottom sheet) and desktop placement as desired. The plugin URL may
accept `?symbol=UNG.US`, but the engine remains locked to the configured symbol
contract.

Required runtime rules:

- Reject unsigned or stale snapshots.
- Render `HOLD`/`INVALIDATED` when the data-quality field is not `valid`.
- Never place trades.
- Never embed cTrader, Telegram, email, or market-data secrets in browser code.
- The page provides a last-updated timestamp and source-quality metadata.
- cTrader’s host session supplies the authenticated market-data bridge; no
  account token is embedded in the page.
