/* Technical-only local engine. No external APIs and no order execution. */
export class LocalHarvestEngine {
  constructor(maxQuotes = 2000) {
    this.maxQuotes = maxQuotes;
    this.quotes = [];
    this.dailyBars = [];
    this.latestQuote = null;
    this.positionState = "hold";
  }

  onDailyBars(payload, digits = 5) {
    const raw = Array.isArray(payload?.bars) ? payload.bars : [];
    const scale = 100000;
    const priceValue = (value) => {
      const n = Number(value);
      return Number.isFinite(n) && Math.abs(n) > 1000 ? n / scale : n;
    };
    const relativeValue = (value) => {
      const n = Number(value);
      return Number.isFinite(n) ? n / scale : 0;
    };
    this.dailyBars = raw.map((bar) => {
      const low = priceValue(bar.low ?? bar.lowPrice);
      const open = bar.open != null ? priceValue(bar.open) : low + relativeValue(bar.deltaOpen ?? 0);
      const high = bar.high != null ? priceValue(bar.high) : low + relativeValue(bar.deltaHigh ?? 0);
      const close = bar.close != null ? priceValue(bar.close) : low + relativeValue(bar.deltaClose ?? 0);
      return { open, high, low, close, timestamp: bar.utcTimestamp ?? bar.timestamp };
    }).filter((bar) => bar.close > 0 && bar.high >= bar.low && bar.close >= bar.low && bar.close <= bar.high);
    return this.evaluate();
  }

  onQuote(quote) {
    const bid = Number(quote.bid ?? quote.bidPrice);
    const ask = Number(quote.ask ?? quote.askPrice);
    const timestamp = quote.timestamp || quote.time || new Date().toISOString();
    if (!(bid > 0) || !(ask >= bid)) return { decision: "INVALIDATED", reason: "invalid bid/ask" };
    this.latestQuote = { mid: (bid + ask) / 2, bid, ask, timestamp: new Date(timestamp).getTime() };
    this.quotes.push(this.latestQuote);
    if (this.quotes.length > this.maxQuotes) this.quotes.shift();
    return this.evaluate();
  }

  evaluate() {
    if (this.dailyBars.length < 30 || !this.latestQuote) return { decision: "WARMING_UP", reason: "waiting for 30 daily cTrader bars and a valid quote", data_quality: "valid" };
    const bars = this.dailyBars.slice(-30);
    const prices = bars.map(q => q.close);
    const price = this.latestQuote.mid;
    const mean = prices.reduce((a, b) => a + b, 0) / prices.length;
    const variance = prices.reduce((a, b) => a + (b - mean) ** 2, 0) / prices.length;
    const sigma = Math.sqrt(variance);
    const upper = Math.min(mean + 2 * sigma, Math.max(...bars.map(b => b.high)));
    const lower = Math.max(mean - 2 * sigma, Math.min(...bars.map(b => b.low)));
    const technicalConfidence = Math.min(1, Math.max(0, 0.5 + (sigma > 0 && Math.abs(price - mean) >= sigma ? 0.15 : 0)));
    const confidence = technicalConfidence * 0.5;
    let decision = "HOLD";
    if (price >= upper && confidence >= 0.30) decision = "SELL_ZONE_CANDIDATE";
    if (price <= lower && confidence >= 0.30) decision = "BUYBACK_ZONE_CANDIDATE";
    return {
      decision, symbol: "UNG.US", price, mean, volatility: sigma,
      sell_zone: [mean, upper], buyback_zone: [lower, mean],
      confidence, fundamental_coverage: 0, data_quality: "valid",
      reason: "technical-only cTrader WebView mode; external fundamentals unavailable",
      generated_at_utc: new Date().toISOString(), data_age_seconds: 0,
    };
  }
}
