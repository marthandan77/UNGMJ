/* Technical-only local engine. No external APIs and no order execution. */
export class LocalHarvestEngine {
  constructor(maxQuotes = 2000) {
    this.maxQuotes = maxQuotes;
    this.quotes = [];
    this.positionState = "hold";
  }

  onQuote(quote) {
    const bid = Number(quote.bid ?? quote.bidPrice);
    const ask = Number(quote.ask ?? quote.askPrice);
    const timestamp = quote.timestamp || quote.time || new Date().toISOString();
    if (!(bid > 0) || !(ask >= bid)) return { decision: "INVALIDATED", reason: "invalid bid/ask" };
    this.quotes.push({ mid: (bid + ask) / 2, bid, ask, timestamp: new Date(timestamp).getTime() });
    if (this.quotes.length > this.maxQuotes) this.quotes.shift();
    return this.evaluate();
  }

  evaluate() {
    if (this.quotes.length < 30) return { decision: "WARMING_UP", reason: "waiting for 30 valid quotes", data_quality: "valid" };
    const prices = this.quotes.map(q => q.mid);
    const price = prices.at(-1);
    const mean = prices.reduce((a, b) => a + b, 0) / prices.length;
    const variance = prices.reduce((a, b) => a + (b - mean) ** 2, 0) / prices.length;
    const sigma = Math.sqrt(variance);
    const upper = mean + 2 * sigma;
    const lower = mean - 2 * sigma;
    const technicalConfidence = Math.min(1, Math.max(0, 0.5 + (sigma > 0 && Math.abs(price - mean) >= sigma ? 0.15 : 0)));
    const confidence = technicalConfidence * 0.5;
    let decision = "HOLD";
    if (price >= upper && confidence >= 0.30) decision = "SELL_ZONE_CANDIDATE";
    if (price <= lower && confidence >= 0.30) decision = "BUYBACK_ZONE_CANDIDATE";
    return {
      decision, symbol: "UNG.US", price, mean, volatility: sigma,
      sell_zone: [mean, upper], buyback_zone: [lower, mean],
      confidence, fundamental_coverage: 0, data_quality: "valid",
      reason: "technical-only local mode; external fundamentals unavailable",
      generated_at_utc: new Date().toISOString(), data_age_seconds: 0,
    };
  }
}
