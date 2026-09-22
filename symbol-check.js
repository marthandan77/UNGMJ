/* Alert-enable gate for the cTrader dashboard. */
export function validateSymbolSnapshot(snapshot) {
  const errors = [];
  if (!snapshot || String(snapshot.symbol || "").toUpperCase() !== "UNG.US") errors.push("symbol is not UNG.US");
  if (!(Number(snapshot.bid) > 0) || !(Number(snapshot.ask) >= Number(snapshot.bid))) errors.push("invalid bid/ask");
  if (!(Number(snapshot.digits) >= 0) || !(Number(snapshot.tick_size) > 0)) errors.push("invalid price metadata");
  if (snapshot.trading_enabled !== true) errors.push("symbol is not trading-enabled");
  if (snapshot.has_daily_history !== true) errors.push("daily history unavailable");
  return { valid: errors.length === 0, errors };
}
