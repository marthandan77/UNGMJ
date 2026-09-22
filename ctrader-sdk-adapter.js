/* cTrader Web Plugin SDK adapter.
 * Documented handshake: confirm -> register -> confirm.
 * This file intentionally reads no account data and exposes no trade methods.
 */
import { createClientAdapter } from "https://esm.sh/@spotware-web-team/sdk-external-api";
import { catchError, take, tap } from "https://esm.sh/rxjs";
import {
  getLightSymbolList,
  getSymbol,
  getTrendbarList,
  handleConfirmEvent,
  quoteEvent,
  registerEvent,
  subscribeQuotes,
} from "https://esm.sh/@spotware-web-team/sdk";

const configuredSymbol = new URLSearchParams(location.search).get("symbol") || "UNG.US";
let adapter;
let connected = false;
let configuredSymbolId = null;
let connectionTimer;
let symbolDetails = {};
let latestQuote = {};
let dailyHistoryAvailable = false;

function fail(message, error) {
  window.dispatchEvent(new CustomEvent("ung-radar-sdk-error", { detail: { message, error: String(error || "") } }));
}

export function connectCTraderHost() {
  adapter = createClientAdapter({ logger: console });
  connectionTimer = setTimeout(() => {
    if (!connected) fail("cTrader host unavailable: open this URL as a registered cTrader WebView plugin, not as a standalone browser page");
  }, 12000);
  handleConfirmEvent(adapter, {}).pipe(take(1)).subscribe();
  registerEvent(adapter).pipe(
    take(1),
    tap(() => {
      handleConfirmEvent(adapter, {}).pipe(take(1)).subscribe();
      connected = true;
      clearTimeout(connectionTimer);
      window.dispatchEvent(new CustomEvent("ung-radar-sdk-connected"));
      subscribeToConfiguredSymbol();
    }),
    catchError(error => {
      fail("cTrader host handshake failed", error);
      return [];
    }),
  ).subscribe();
}

function subscribeToConfiguredSymbol() {
  getLightSymbolList(adapter, {}).pipe(
    take(1),
    tap(response => {
      const symbols = response.symbols || response.symbol || [];
      const match = symbols.find(item => String(item.symbolName || item.name || "").toUpperCase() === configuredSymbol.toUpperCase());
      if (!match || match.symbolId == null) {
        fail(`Configured symbol not found: ${configuredSymbol}`);
        return;
      }
      configuredSymbolId = match.symbolId;
      symbolDetails = match;
      getSymbol(adapter, { symbolId: [configuredSymbolId] }).pipe(
        take(1),
        tap(detailResponse => {
          const details = detailResponse.symbol || detailResponse.symbols || [];
          symbolDetails = { ...match, ...(details[0] || {}) };
          emitMetadata();
        }),
        catchError(error => {
          fail("Detailed symbol metadata request failed", error);
          return [];
        }),
      ).subscribe();
      requestDailyBars();
      subscribeQuotes(adapter, { symbolId: [match.symbolId], subscribeToSpotTimestamp: true }).pipe(take(1)).subscribe({
        error: error => fail("Quote subscription failed", error),
      });
    }),
    catchError(error => {
      fail("Symbol list request failed", error);
      return [];
    }),
  ).subscribe();

  quoteEvent(adapter).pipe(
    tap(quote => {
      const quoteSymbolId = quote.symbolId ?? quote.symbol_id;
      if (configuredSymbolId != null && quoteSymbolId != null && Number(quoteSymbolId) !== Number(configuredSymbolId)) return;
      latestQuote = quote;
      emitMetadata();
      window.dispatchEvent(new CustomEvent("ung-radar-quote", { detail: quote }));
    }),
    catchError(error => {
      fail("Quote stream failed", error);
      return [];
    }),
  ).subscribe();
}

function requestDailyBars() {
  if (configuredSymbolId == null) return;
  getTrendbarList(adapter, {
    symbolId: configuredSymbolId,
    period: "D1",
    count: 30,
    toTimestamp: Date.now(),
    type: "REGULAR",
  }).pipe(
    take(1),
    tap(response => {
      const bars = response.trendbar || response.trendbars || [];
      dailyHistoryAvailable = bars.length >= 30;
      emitMetadata();
      window.dispatchEvent(new CustomEvent("ung-radar-daily-bars", { detail: {
        symbol_id: configuredSymbolId,
        bars,
        received_at_utc: new Date().toISOString(),
      }}));
    }),
    catchError(error => {
      fail("Daily trendbar request failed", error);
      return [];
    }),
  ).subscribe();
}

function emitMetadata() {
  const digits = Number(symbolDetails.digits ?? symbolDetails.precision);
  const tickSize = Number(symbolDetails.tickSize ?? symbolDetails.tick_size ?? symbolDetails.minChange);
  const bid = Number(latestQuote.bid ?? latestQuote.bidPrice ?? symbolDetails.bid);
  const ask = Number(latestQuote.ask ?? latestQuote.askPrice ?? symbolDetails.ask);
  window.dispatchEvent(new CustomEvent("ung-radar-symbol-metadata", { detail: {
    symbol: String(symbolDetails.symbolName || symbolDetails.name || configuredSymbol),
    symbol_id: configuredSymbolId,
    bid,
    ask,
    digits,
    tick_size: tickSize,
    trading_enabled: symbolDetails.tradingEnabled ?? symbolDetails.trading_enabled ?? symbolDetails.enabled !== false,
    has_daily_history: dailyHistoryAvailable,
    metadata_received_at_utc: new Date().toISOString(),
  }}));
}

export function isCTraderConnected() {
  return connected;
}
