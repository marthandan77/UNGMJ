/* cTrader Web Plugin SDK adapter.
 * Documented handshake: confirm -> register -> confirm.
 * This file intentionally reads no account data and exposes no trade methods.
 */
import { createClientAdapter } from "https://esm.sh/@spotware-web-team/sdk-external-api";
import { catchError, take, tap } from "https://esm.sh/rxjs";
import {
  getLightSymbolList,
  handleConfirmEvent,
  quoteEvent,
  registerEvent,
  subscribeQuotes,
} from "https://esm.sh/@spotware-web-team/sdk";

const configuredSymbol = new URLSearchParams(location.search).get("symbol") || "UNG.US";
let adapter;
let connected = false;
let configuredSymbolId = null;

function fail(message, error) {
  window.dispatchEvent(new CustomEvent("ung-radar-sdk-error", { detail: { message, error: String(error || "") } }));
}

export function connectCTraderHost() {
  adapter = createClientAdapter({ logger: console });
  handleConfirmEvent(adapter, {}).pipe(take(1)).subscribe();
  registerEvent(adapter).pipe(
    take(1),
    tap(() => {
      handleConfirmEvent(adapter, {}).pipe(take(1)).subscribe();
      connected = true;
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
      window.dispatchEvent(new CustomEvent("ung-radar-symbol-metadata", { detail: {
        symbol: String(match.symbolName || match.name || configuredSymbol),
        symbol_id: match.symbolId,
        bid: Number(match.bid ?? match.bidPrice),
        ask: Number(match.ask ?? match.askPrice),
        digits: Number(match.digits),
        tick_size: Number(match.tickSize ?? match.tick_size),
        trading_enabled: match.tradingEnabled ?? match.trading_enabled,
        has_daily_history: match.hasDailyHistory ?? match.has_daily_history,
        metadata_received_at_utc: new Date().toISOString(),
      }}));
      subscribeQuotes(adapter, { symbolId: [match.symbolId] }).pipe(take(1)).subscribe({
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
      window.dispatchEvent(new CustomEvent("ung-radar-quote", { detail: quote }));
    }),
    catchError(error => {
      fail("Quote stream failed", error);
      return [];
    }),
  ).subscribe();
}

export function isCTraderConnected() {
  return connected;
}
