// GTA VI trophy rarity forecaster — dilution-aware, hierarchical Theil-Sen.
//
// Model, in one paragraph: Exophase's page is a staircase — flat between its
// own refreshes, jumping when it refreshes — so the history is first collapsed
// to distinct states (consecutive identical reads merged). For each trophy the
// rate is split into earners and players-tracked, both fitted in log space
// against log time with a Theil-Sen slope (robust to refresh jumps), the
// trophy slope shrunk toward its tier mean, and the forecast rate rebuilt as
// earners / tracked. A direct logit(rate) trend fit cross-checks that number.
// When fewer than 3 distinct states exist no trend can be fitted, so a
// dilution prior is used instead: the next refresh grows tracked players much
// faster than earners, so rates fall. Quantiles come from a normal spread on
// the logit scale. Node 22, standard library only, no network, deterministic.
//
// Every tunable number below is a named constant with a one-line reason; no
// bare magic numbers are used in the model itself.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

/** Quantile levels the contest scores, in output order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard-normal quantiles at LEVELS. */
const Z_SCORES = [-1.645, -0.674, 0, 0.674, 1.645];

/** Smallest / largest rate (percent) fed to a logit; the scorer clamps the same way. */
const RATE_FLOOR = 0.01;
const RATE_CEIL = 99.99;
/** Half an earner added before taking logs, so a 0.00% rate stays finite. */
const EARNERS_SMOOTH = 0.5;

/** Distinct states needed before a trend is fitted at all. */
const MIN_STATES_FOR_TREND = 3;
/** Pseudo-count (in states) pulling a trophy's slope toward its tier mean. */
const TIER_SHRINK_STRENGTH = 4;
/** Weight of the counts-based forecast when it disagrees with the direct logit fit. */
const COUNTS_BLEND_WEIGHT = 0.6;
/** Logit gap above which the two forecasts are blended instead of trusting counts alone. */
const BLEND_GAP_LOGIT = 0.5;
/** Damping: a trend extrapolated far past the observed log-time span is pulled
    back toward the last rate (damped-trend idea: far extrapolations saturate). */
const DAMPING_ENABLED = true;

/** Prior tracked-player growth over one 90-minute horizon in the no-trend regime. */
const PRIOR_N_GROWTH_90MIN = 4.8;
/** Prior earner growth over one 90-minute horizon in the no-trend regime. */
const PRIOR_EARNERS_GROWTH_90MIN = 1.25;
/** Caps on the prior growth factors, so a long horizon cannot explode them. */
const PRIOR_N_GROWTH_CAP = 12;
const PRIOR_EARNERS_GROWTH_CAP = 2.5;
/** Spread (logit) used when no trend exists: wide, because a refresh may or may not land. */
const PRIOR_SPREAD = 1.0;

/** Random-walk spread per sqrt-day of horizon (same scale the house baseline uses). */
const HORIZON_SIGMA = 0.5;
/** Extra logit variance allowed for refresh timing in the fitted regime. */
const REFRESH_VAR_SCALE = 0.25;
/** Fallback refresh variance when too few states exist to estimate it. */
const REFRESH_VAR_FALLBACK = 0.64;
/** Smallest / largest logit spread handed to the quantiles. */
const SPREAD_FLOOR = 0.35;
const SPREAD_CAP = 1.6;

/** A rate at/above this is treated as sticky at ~100% (it did not fall at refresh). */
const STICKY_RATE = 99.99;
/** Median and spread for a sticky 100% trophy. */
const STICKY_MEDIAN = 99.7;
const STICKY_SPREAD = 0.12;
/** Largest plausible tracked-player growth per fitted extrapolation (sanity cap). */
const MAX_N_GROWTH = 50;

interface TrophyMeta {
  id: number;
  tier: string;
}
interface HistoryRead {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface CaseFile {
  levels: number[];
  horizon_minutes: number;
  target: string;
  trophies: TrophyMeta[];
  history: HistoryRead[];
}
interface DistinctState {
  timeMs: number;
  players: number;
  rates: Array<number | null>;
}

const clampRate = (x: number): number =>
  Math.min(RATE_CEIL, Math.max(RATE_FLOOR, x));
const logitOf = (pct: number): number => {
  const r = clampRate(pct);
  return Math.log(r / (100 - r));
};
const logisticPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const medianOf = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) return 0;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};
const meanOf = (xs: number[]): number =>
  xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
const varianceOf = (xs: number[]): number => {
  if (xs.length < 2) return 0;
  const m = meanOf(xs);
  return xs.reduce((a, x) => a + (x - m) * (x - m), 0) / (xs.length - 1);
};

/** Median pairwise slope; 0 when fewer than two points (no trend identifiable). */
function theilSenSlope(xs: number[], ys: number[]): number {
  const slopes: number[] = [];
  for (let i = 0; i < xs.length; i++) {
    for (let j = i + 1; j < xs.length; j++) {
      const dx = xs[j] - xs[i];
      if (dx !== 0) slopes.push((ys[j] - ys[i]) / dx);
    }
  }
  return slopes.length === 0 ? 0 : medianOf(slopes);
}

/** Median intercept of y - slope*x, so the line passes through the data's middle. */
function theilSenIntercept(xs: number[], ys: number[], slope: number): number {
  return medianOf(xs.map((x, i) => ys[i] - slope * x));
}

/** Collapse consecutive reads with identical N and rates into one state. */
function distinctStates(history: HistoryRead[]): DistinctState[] {
  const out: DistinctState[] = [];
  for (const read of history) {
    const state: DistinctState = {
      timeMs: Date.parse(read.at),
      players: read.players_tracked,
      rates: read.rates,
    };
    const prev = out[out.length - 1];
    const same =
      prev !== undefined &&
      prev.players === state.players &&
      prev.rates.length === state.rates.length &&
      prev.rates.every((r, i) => r === state.rates[i]);
    if (same) continue;
    out.push(state);
  }
  return out;
}

/** Hours since the first distinct state, in log time (launch growth is front-loaded). */
function logTimeOf(timeMs: number, anchorMs: number): number {
  const hours = Math.max(0, (timeMs - anchorMs) / 3_600_000);
  return Math.log1p(hours);
}

function forecastCase(c: CaseFile): Array<{ id: number; q: number[] }> {
  const nTrophies = c.trophies.length;
  const states = distinctStates(c.history);
  const last = states[states.length - 1];
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;
  const horizonScale = Math.sqrt(Math.max(0, c.horizon_minutes) / 90);

  // Last non-null rate per trophy (falls back to the tier's last median, then 1%).
  const lastRate: number[] = c.trophies.map((_, i) => {
    for (let k = states.length - 1; k >= 0; k--) {
      const r = states[k].rates[i];
      if (r !== null && r !== undefined) return r;
    }
    return NaN;
  });
  const tierLastMedian = new Map<string, number>();
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const vals = c.trophies
      .map((t, i) => (t.tier === tier && Number.isFinite(lastRate[i]) ? lastRate[i] : NaN))
      .filter((v) => Number.isFinite(v));
    tierLastMedian.set(tier, vals.length > 0 ? medianOf(vals) : 1);
  }
  for (let i = 0; i < nTrophies; i++) {
    if (!Number.isFinite(lastRate[i])) {
      lastRate[i] = tierLastMedian.get(c.trophies[i].tier) ?? 1;
    }
  }

  // ---- No-trend regime: fewer than 3 distinct states -> dilution prior. ----
  if (states.length < MIN_STATES_FOR_TREND || last === undefined) {
    const nGrowth = Math.min(
      PRIOR_N_GROWTH_CAP,
      Math.exp(Math.log(PRIOR_N_GROWTH_90MIN) * horizonScale),
    );
    const eGrowth = Math.min(
      PRIOR_EARNERS_GROWTH_CAP,
      Math.exp(Math.log(PRIOR_EARNERS_GROWTH_90MIN) * horizonScale),
    );
    return c.trophies.map((t, i) => {
      if (lastRate[i] >= STICKY_RATE) {
        return { id: t.id, q: quantiles(STICKY_MEDIAN, STICKY_SPREAD) };
      }
      const medianPct = clampRate(lastRate[i] * (eGrowth / nGrowth));
      return { id: t.id, q: quantiles(medianPct, PRIOR_SPREAD) };
    });
  }

  // ---- Fitted regime: Theil-Sen on log earners and log tracked vs log time. ----
  const anchorMs = states[0].timeMs;
  const xs = states.map((s) => logTimeOf(s.timeMs, anchorMs));
  const targetMs = Date.parse(c.target);
  const xTarget = logTimeOf(
    Number.isFinite(targetMs) ? targetMs : last.timeMs,
    anchorMs,
  );

  // Tracked players: one shared fit for all trophies.
  const logN = states.map((s) => Math.log(Math.max(1, s.players)));
  const nSlope = theilSenSlope(xs, logN);
  const nIntercept = theilSenIntercept(xs, logN, nSlope);
  const logNTarget = nIntercept + nSlope * xTarget;
  const nForecast = Math.min(
    last.players * MAX_N_GROWTH,
    Math.max(last.players, Math.exp(logNTarget)),
  );
  // Refresh uncertainty: how jumpy the tracked count is between distinct states.
  const logNIncrements: number[] = [];
  for (let k = 1; k < states.length; k++) {
    logNIncrements.push(logN[k] - logN[k - 1]);
  }
  const refreshVar =
    logNIncrements.length >= 2
      ? REFRESH_VAR_SCALE * varianceOf(logNIncrements)
      : REFRESH_VAR_FALLBACK;

  // Per-trophy raw slopes first, so tier means can be computed for shrinkage.
  interface TrophyFit {
    slope: number;
    intercept: number;
    count: number;
    residVar: number;
  }
  const rawFits: Array<TrophyFit | null> = c.trophies.map((_, i) => {
    const px: number[] = [];
    const py: number[] = [];
    states.forEach((s, k) => {
      const r = s.rates[i];
      if (r === null || r === undefined) return;
      const earners = (clampRate(r) / 100) * s.players + EARNERS_SMOOTH;
      px.push(xs[k]);
      py.push(Math.log(earners));
    });
    if (px.length < 2) return null;
    const slope = theilSenSlope(px, py);
    const intercept = theilSenIntercept(px, py, slope);
    const resid = px.map((x, j) => py[j] - (intercept + slope * x));
    return { slope, intercept, count: px.length, residVar: varianceOf(resid) };
  });

  // Tier mean slope (over trophies with a fit), plus a global fallback.
  const tierSlopeMean = new Map<string, number>();
  const allSlopes = rawFits
    .filter((f): f is TrophyFit => f !== null)
    .map((f) => f.slope);
  const globalSlope = allSlopes.length > 0 ? meanOf(allSlopes) : 0;
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const vals = rawFits
      .map((f, i) => (c.trophies[i].tier === tier && f !== null ? f.slope : NaN))
      .filter((v) => Number.isFinite(v));
    tierSlopeMean.set(tier, vals.length > 0 ? meanOf(vals) : globalSlope);
  }

  return c.trophies.map((t, i) => {
    if (lastRate[i] >= STICKY_RATE) {
      return { id: t.id, q: quantiles(STICKY_MEDIAN, STICKY_SPREAD) };
    }
    const fit = rawFits[i];
    let medianLogit: number;
    let residVar = 0;
    if (fit === null) {
      // No usable earners series: fall back to the last rate, wide spread.
      medianLogit = logitOf(lastRate[i]);
      residVar = REFRESH_VAR_FALLBACK;
    } else {
      // Shrink this trophy's slope toward its tier mean (hierarchical).
      const tierMean = tierSlopeMean.get(t.tier) ?? globalSlope;
      const slope =
        (fit.count * fit.slope + TIER_SHRINK_STRENGTH * tierMean) /
        (fit.count + TIER_SHRINK_STRENGTH);
      const intercept = theilSenIntercept(
        states
          .map((s, k) => ({ s, k }))
          .filter(({ s }) => s.rates[i] !== null && s.rates[i] !== undefined)
          .map(({ k }) => xs[k]),
        states
          .filter((s) => s.rates[i] !== null && s.rates[i] !== undefined)
          .map((s) => Math.log((clampRate(s.rates[i] as number) / 100) * s.players + EARNERS_SMOOTH)),
        slope,
      );
      const earnersForecast = Math.exp(intercept + slope * xTarget);
      const rateFromCounts = clampRate((earnersForecast / nForecast) * 100);
      medianLogit = logitOf(rateFromCounts);
      residVar = fit.residVar;

      // Cross-check: direct logit(rate) trend; blend when the two disagree.
      const qx: number[] = [];
      const qy: number[] = [];
      states.forEach((s, k) => {
        const r = s.rates[i];
        if (r === null || r === undefined) return;
        qx.push(xs[k]);
        qy.push(logitOf(r));
      });
      if (qx.length >= 2) {
        const qSlope = theilSenSlope(qx, qy);
        const qIntercept = theilSenIntercept(qx, qy, qSlope);
        const directLogit = qIntercept + qSlope * xTarget;
        if (Math.abs(directLogit - medianLogit) > BLEND_GAP_LOGIT) {
          medianLogit =
            COUNTS_BLEND_WEIGHT * medianLogit +
            (1 - COUNTS_BLEND_WEIGHT) * directLogit;
        }
      }
    }
    const spread = Math.min(
      SPREAD_CAP,
      Math.max(
        SPREAD_FLOOR,
        Math.sqrt(
          (HORIZON_SIGMA * Math.sqrt(horizonDays)) ** 2 + residVar + refreshVar,
        ),
      ),
    );
    // Damp the trend toward persistence: the further the target lies past the
    // observed span IN LINEAR TIME, the less a fitted slope is trusted
    // (player growth and completion rates saturate; log-time compresses that
    // distance, so the weight uses the linear-time ratio, not log-time).
    let finalLogit = medianLogit;
    if (DAMPING_ENABLED && fit !== null) {
      const spanLin = last.timeMs - states[0].timeMs;
      const gapLin = Math.max(
        0,
        (Number.isFinite(targetMs) ? targetMs : last.timeMs) - last.timeMs,
      );
      if (spanLin > 0 && gapLin > 0) {
        const trendWeight = spanLin / (spanLin + gapLin);
        finalLogit =
          trendWeight * medianLogit + (1 - trendWeight) * logitOf(lastRate[i]);
      }
    }
    return { id: t.id, q: quantiles(logisticPct(finalLogit), spread) };
  });
}

/** Five quantiles around a median percent with a logit-normal spread. */
function quantiles(medianPct: number, spread: number): number[] {
  const centre = logitOf(medianPct);
  const qs = Z_SCORES.map((z) => round4(logisticPct(centre + z * spread)));
  // Rounding can tie or (never) invert neighbours; enforce non-decreasing.
  for (let j = 1; j < qs.length; j++) {
    if (qs[j] < qs[j - 1]) qs[j] = qs[j - 1];
  }
  return qs.map((q) => Math.min(100, Math.max(0, q)));
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith(".in"))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as CaseFile;
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((t, j) => t !== LEVELS[j])
  ) {
    throw new Error(`${file} asks for different levels`);
  }
  if (!Array.isArray(c.history) || c.history.length === 0) {
    throw new Error(`${file} has no history`);
  }
  const trophies = forecastCase(c);
  writeFileSync(
    join(OUTPUT_DIR, `${id}.out`),
    `${JSON.stringify({ trophies })}\n`,
  );
}
