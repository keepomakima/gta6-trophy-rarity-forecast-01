// GTA VI trophy rarity forecaster v2 — saturating hyperbola + logistic N +
// LOO-blended candidates + recalibrated spread.
//
// Model, in one paragraph: Exophase's page is a staircase — flat between its
// own refreshes, jumping when it refreshes — so the history is first collapsed
// to distinct states (consecutive identical reads merged). For each trophy the
// rate is modelled three ways and the three candidates are blended with
// weights from expanding-window forward validation (softmax over each
// candidate's median |logit error| when predicting later states from
// earlier ones only): (a) a saturating hyperbola r(N) = p + c/N
// fitted by Theil-Sen of rate against 1/N — the exact functional form when a
// fixed share p of new players earns the trophy, so the rate saturates at p
// as the tracked population grows; (b) the v1 counts model, Theil-Sen on
// log(earners) and log(tracked) against log time; (c) a direct Theil-Sen fit
// of logit(rate) against log time. The population at the target comes from a
// logistic N(t) fit (grid-searched, SSE in log N), falling back to the v1
// log-time N fit when there are too few states. When fewer than 3 distinct
// states exist no trend can be fitted, so a dilution prior is used instead:
// the next refresh grows tracked players much faster than earners, so rates
// fall. Quantiles come from a normal spread on the logit scale whose size is
// driven by the LOO residual, not a fixed guess. Node 22, standard library
// only, no network, deterministic.
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
/** Distinct states needed before the logistic N fit is attempted. */
const MIN_STATES_FOR_LOGISTIC = 4;
/** Pseudo-count (in states) pulling a trophy's slope/asymptote toward its tier mean. */
const TIER_SHRINK_STRENGTH = 4;
/** Pseudo-count (in states) shrinking the hyperbola slope c toward 0 (persistence). */
const HYPERBOLA_SHRINK_STRENGTH = 4;
/** Softmax temperature for LOO blend weights. Calibrated on Uji 3/4: 0.3 left
    the weights near-uniform whenever candidate errors differed by <0.2 logit
    — exactly the range where the saturating and trend models differ in
    validation while diverging by >1 logit at the target — so the blend
    failed to select; 0.1 selects (0.05 gave no further gain). */
const LOO_TEMPERATURE = 0.1;
/** Split fractions for expanding-window forward validation (train on the
    first fraction of states, predict the rest — tests long-range extrapolation).
    Calibrated: only the earliest split (0.35) is used, because later splits
    predict too short a range and dilute the score with near predictions
    where every candidate looks good. */
const LOO_SPLIT_FRACTIONS = [0.35];
/** Blend weights (hyperbola / counts / direct-logit) when too few states for LOO. */
const LOO_FALLBACK_WEIGHTS = [0.5, 0.3, 0.2];
/** Damping of the fitted trend toward the last rate in linear time. Kept:
    Uji 3 and Uji 4 both prove it helps (Uji 3 0.1432 -> 0.1351, Uji 4
    0.2003 -> 0.1624 at SIGMA_FIT 0.1), because the blended median still
    overshoots when the trend candidates run away; the risk is undershoot
    if the real final rises faster than both synthetic regimes. */
const DAMPING_ENABLED = true;
/** Convert a median absolute logit error to a Gaussian sigma (0.6745 = z at 0.75). */
const MEDIAN_TO_SIGMA = 0.6745;

/** Prior tracked-player growth over one 90-minute horizon in the no-trend regime. */
const PRIOR_N_GROWTH_90MIN = 6.0;
/** Prior earner growth over one 90-minute horizon in the no-trend regime. */
const PRIOR_EARNERS_GROWTH_90MIN = 1.3;
/** Caps on the prior growth factors, so a long horizon cannot explode them. */
const PRIOR_N_GROWTH_CAP = 12;
const PRIOR_EARNERS_GROWTH_CAP = 2.5;
/** Spread (logit) used when no trend exists: refresh may or may not land. */
const PRIOR_SPREAD = 0.55;

/** Random-walk spread per sqrt-day of horizon. Calibrated on Uji 2/3/4:
    sweep 0.25/0.2/0.15/0.1 gave Uji 3 0.190/0.172/0.155/0.143 and Uji 4
    0.235/0.219/0.207/0.200; 0.05 and 0.0 were only marginally better
    (0.140/0.197) and would claim ~zero horizon risk, so 0.1 is kept. */
const SIGMA_FIT = 0.1;
/** Extra logit variance allowed for refresh timing in the fitted regime. */
const REFRESH_VAR_SCALE = 0.25;
/** Fallback refresh variance when too few states exist to estimate it. */
const REFRESH_VAR_FALLBACK = 0.64;
/** Smallest / largest logit spread handed to the quantiles. */
const SPREAD_FLOOR = 0.3;
const SPREAD_CAP = 1.2;

/** A rate at/above this is treated as sticky at ~100% (it did not fall at refresh). */
const STICKY_RATE = 99.99;
/** Median and spread for a sticky 100% trophy. */
const STICKY_MEDIAN = 99.7;
const STICKY_SPREAD = 0.12;
/** Largest plausible tracked-player growth per fitted extrapolation (sanity cap). */
const MAX_N_GROWTH = 50;
/** Lowest plausible forecast rate, as a fraction of the last rate (never below 0.01). */
const RATE_FLOOR_FACTOR = 0.15;

/** Logistic N grid: carrying-capacity multiples of the last tracked count. */
const LOGISTIC_K_MULTIPLIERS = [2, 3, 5, 8, 12, 20, 50, 100];
/** Logistic N grid: absolute carrying capacities (launch population is in the 100ks). */
const LOGISTIC_K_ABSOLUTE = [1_000_000, 2_000_000];
/** Logistic N grid: growth rates per day (spec values plus intermediates for slow growth). */
const LOGISTIC_RATES = [0.5, 0.7, 0.9, 1.1, 1.4, 1.7, 2, 2.5, 3, 4, 5, 6];

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
interface LineFit {
  slope: number;
  intercept: number;
}
interface LogisticFit {
  k: number;
  r: number;
  a: number;
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

/** Theil-Sen line fit; null when fewer than two points or all x identical. */
function theilSenFit(xs: number[], ys: number[]): LineFit | null {
  if (xs.length < 2) return null;
  const slopes: number[] = [];
  for (let i = 0; i < xs.length; i++) {
    for (let j = i + 1; j < xs.length; j++) {
      const dx = xs[j] - xs[i];
      if (dx !== 0) slopes.push((ys[j] - ys[i]) / dx);
    }
  }
  if (slopes.length === 0) return null;
  const slope = medianOf(slopes);
  const intercept = medianOf(xs.map((x, i) => ys[i] - slope * x));
  return { slope, intercept };
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

/** Days since the first distinct state (linear time, for the logistic N fit). */
function daysSince(timeMs: number, anchorMs: number): number {
  return Math.max(0, (timeMs - anchorMs) / 86_400_000);
}

/**
 * Fit N(t) = K / (1 + A*exp(-r*t)) over the distinct states by deterministic
 * grid search on (K, r), minimising SSE in log N; A follows from N(0).
 * Null when too few states or no grid point is usable.
 */
function fitLogisticN(states: DistinctState[]): LogisticFit | null {
  if (states.length < MIN_STATES_FOR_LOGISTIC) return null;
  const anchorMs = states[0].timeMs;
  const n0 = states[0].players;
  const lastN = states[states.length - 1].players;
  const ts = states.map((s) => daysSince(s.timeMs, anchorMs));
  const logNs = states.map((s) => Math.log(Math.max(1, s.players)));
  const kCandidates = [
    ...LOGISTIC_K_MULTIPLIERS.map((m) => lastN * m),
    ...LOGISTIC_K_ABSOLUTE,
  ];
  let best: LogisticFit | null = null;
  let bestSse = Infinity;
  for (const k of kCandidates) {
    if (k <= n0) continue;
    const a = k / n0 - 1;
    for (const r of LOGISTIC_RATES) {
      let sse = 0;
      for (let i = 0; i < states.length; i++) {
        const pred = Math.log(k / (1 + a * Math.exp(-r * ts[i])));
        const d = logNs[i] - pred;
        sse += d * d;
      }
      if (sse < bestSse) {
        bestSse = sse;
        best = { k, r, a };
      }
    }
  }
  return best;
}

/** Logistic N at t days after the anchor, from a fitted (K, r, A). */
function logisticNAt(fit: LogisticFit, tDays: number): number {
  return fit.k / (1 + fit.a * Math.exp(-fit.r * tDays));
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
  // RULES.txt line 11 documents the first refresh: N 638 -> 3,868 (x6.06),
  // so the prior N growth is 6.0, not v1's 4.8; earners grow only ~x1.3
  // (most of the new tracked players have not earned the trophy yet).
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

  // ---- Fitted regime. ----
  const anchorMs = states[0].timeMs;
  const xs = states.map((s) => logTimeOf(s.timeMs, anchorMs));
  const targetMs = Date.parse(c.target);
  const targetTimeMs = Number.isFinite(targetMs) ? targetMs : last.timeMs;
  const xTarget = logTimeOf(targetTimeMs, anchorMs);
  const tTargetDays = daysSince(targetTimeMs, anchorMs);

  // Tracked players at target: logistic fit, else the v1 log-time fit.
  const logN = states.map((s) => Math.log(Math.max(1, s.players)));
  const logNFit = theilSenFit(xs, logN);
  const nFromLogTime =
    logNFit === null
      ? last.players
      : Math.exp(logNFit.intercept + logNFit.slope * xTarget);
  const logisticFit = fitLogisticN(states);
  const nForecastLogistic =
    logisticFit === null
      ? null
      : Math.min(
          logisticFit.k,
          last.players * MAX_N_GROWTH,
          Math.max(last.players, logisticNAt(logisticFit, tTargetDays)),
        );
  // Candidate (b) keeps the log-time N; candidate (a) uses the logistic N.
  const nForecastCounts = Math.min(
    last.players * MAX_N_GROWTH,
    Math.max(last.players, nFromLogTime),
  );
  const nForecastHyperbola = nForecastLogistic ?? nForecastCounts;

  // Refresh uncertainty: how jumpy the tracked count is between distinct states.
  const logNIncrements: number[] = [];
  for (let k = 1; k < states.length; k++) {
    logNIncrements.push(logN[k] - logN[k - 1]);
  }
  const refreshVar =
    logNIncrements.length >= 2
      ? REFRESH_VAR_SCALE * varianceOf(logNIncrements)
      : REFRESH_VAR_FALLBACK;

  // Per-trophy raw fits, so tier means can be computed for shrinkage.
  interface HyperbolaFit {
    p: number;
    c: number;
    count: number;
  }
  interface CountsFit {
    slope: number;
    intercept: number;
    count: number;
  }
  const rawHyperbola: Array<HyperbolaFit | null> = c.trophies.map((_, i) => {
    const hx: number[] = [];
    const hy: number[] = [];
    states.forEach((s) => {
      const r = s.rates[i];
      if (r === null || r === undefined) return;
      hx.push(1 / Math.max(1, s.players));
      hy.push(r);
    });
    const fit = theilSenFit(hx, hy);
    if (fit === null) return null;
    // r(N) = p + c/N: intercept is the asymptote, slope is c.
    return { p: fit.intercept, c: fit.slope, count: hx.length };
  });
  const rawCounts: Array<CountsFit | null> = c.trophies.map((_, i) => {
    const px: number[] = [];
    const py: number[] = [];
    states.forEach((s, k) => {
      const r = s.rates[i];
      if (r === null || r === undefined) return;
      const earners = (clampRate(r) / 100) * s.players + EARNERS_SMOOTH;
      px.push(xs[k]);
      py.push(Math.log(earners));
    });
    const fit = theilSenFit(px, py);
    if (fit === null) return null;
    return { slope: fit.slope, intercept: fit.intercept, count: px.length };
  });

  // Tier means for shrinkage (hyperbola asymptote; counts slope), plus global fallbacks.
  const tierMeanP = new Map<string, number>();
  const allP = rawHyperbola
    .filter((f): f is HyperbolaFit => f !== null)
    .map((f) => clampRate(f.p));
  const globalP = allP.length > 0 ? meanOf(allP) : 1;
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const vals = rawHyperbola
      .map((f, i) => (c.trophies[i].tier === tier && f !== null ? clampRate(f.p) : NaN))
      .filter((v) => Number.isFinite(v));
    tierMeanP.set(tier, vals.length > 0 ? meanOf(vals) : globalP);
  }
  const tierSlopeMean = new Map<string, number>();
  const allSlopes = rawCounts
    .filter((f): f is CountsFit => f !== null)
    .map((f) => f.slope);
  const globalSlope = allSlopes.length > 0 ? meanOf(allSlopes) : 0;
  for (const tier of new Set(c.trophies.map((t) => t.tier))) {
    const vals = rawCounts
      .map((f, i) => (c.trophies[i].tier === tier && f !== null ? f.slope : NaN))
      .filter((v) => Number.isFinite(v));
    tierSlopeMean.set(tier, vals.length > 0 ? meanOf(vals) : globalSlope);
  }

  // Shrunk hyperbola forecast for one trophy (full-data fit).
  const hyperbolaRate = (i: number): number | null => {
    const fit = rawHyperbola[i];
    if (fit === null) return null;
    const tierMean = tierMeanP.get(c.trophies[i].tier) ?? globalP;
    const n = fit.count;
    const pShrunk =
      (n * clampRate(fit.p) + TIER_SHRINK_STRENGTH * tierMean) /
      (n + TIER_SHRINK_STRENGTH);
    const cShrunk =
      (fit.c * n) / (n + HYPERBOLA_SHRINK_STRENGTH);
    const r = pShrunk + cShrunk / nForecastHyperbola;
    return clampRate(
      Math.max(r, Math.max(RATE_FLOOR, lastRate[i] * RATE_FLOOR_FACTOR)),
    );
  };

  // Shrunk counts-model forecast for one trophy (full-data fit).
  const countsRate = (i: number): number | null => {
    const fit = rawCounts[i];
    if (fit === null || logNFit === null) return null;
    const tierMean = tierSlopeMean.get(c.trophies[i].tier) ?? globalSlope;
    const slope =
      (fit.count * fit.slope + TIER_SHRINK_STRENGTH * tierMean) /
      (fit.count + TIER_SHRINK_STRENGTH);
    // Recompute the intercept at the shrunk slope so the line stays centred.
    const px: number[] = [];
    const py: number[] = [];
    states.forEach((s, k) => {
      const r = s.rates[i];
      if (r === null || r === undefined) return;
      px.push(xs[k]);
      py.push(Math.log((clampRate(r) / 100) * s.players + EARNERS_SMOOTH));
    });
    const intercept = medianOf(px.map((x, j) => py[j] - slope * x));
    const earnersForecast = Math.exp(intercept + slope * xTarget);
    return clampRate((earnersForecast / nForecastCounts) * 100);
  };

  // Direct logit-trend forecast for one trophy (full-data fit).
  const directRate = (i: number): number | null => {
    const qx: number[] = [];
    const qy: number[] = [];
    states.forEach((s, k) => {
      const r = s.rates[i];
      if (r === null || r === undefined) return;
      qx.push(xs[k]);
      qy.push(logitOf(r));
    });
    const fit = theilSenFit(qx, qy);
    if (fit === null) return null;
    return clampRate(logisticPct(fit.intercept + fit.slope * xTarget));
  };

  return c.trophies.map((t, i) => {
    if (lastRate[i] >= STICKY_RATE) {
      return { id: t.id, q: quantiles(STICKY_MEDIAN, STICKY_SPREAD) };
    }
    const candA = hyperbolaRate(i) ?? lastRate[i];
    const candB = countsRate(i) ?? lastRate[i];
    const candC = directRate(i) ?? lastRate[i];
    const candidates = [candA, candB, candC];

    // ---- Expanding-window forward validation (leave-future-out): split the
    // history at the earliest fraction, refit each candidate on the states
    // before the split only, and score it on the FARTHEST later states.
    // This tests the exact skill the target forecast needs — extrapolating
    // days ahead — which interpolation LOO and one-step-ahead LOO cannot:
    // on a staircase every candidate interpolates well, and one refresh
    // ahead every trend looks fine; only a long-range forward test exposes
    // a log-time trend that runs away while the truth saturates. (Measured
    // on Uji 3/4: the hyperbola's error shrinks with prediction distance
    // while the trend models' grows, so the median over all later states
    // hid the difference and the farthest-3 score reveals it.)
    // Per-candidate median |logit error| over those far states, blended
    // with softmax(-err/0.1). ----
    let weights = LOO_FALLBACK_WEIGHTS;
    let looBlendErr = 0;
    if (states.length >= MIN_STATES_FOR_LOGISTIC) {
      const errs: number[][] = [[], [], []];
      const splitPoints = new Set<number>();
      for (const f of LOO_SPLIT_FRACTIONS) {
        const m = Math.max(
          MIN_STATES_FOR_TREND,
          Math.floor(f * states.length),
        );
        if (m < states.length - 1) splitPoints.add(m);
      }
      for (const m of [...splitPoints].sort((a, b) => a - b)) {
        const kept = states.slice(0, m);
        const keptXs = kept.map((s) => logTimeOf(s.timeMs, anchorMs));
        // (a) hyperbola refit on the prefix (shrunk once, reused below).
        let hyperParams: { p: number; c: number } | null = null;
        {
          const hx: number[] = [];
          const hy: number[] = [];
          kept.forEach((s) => {
            const r = s.rates[i];
            if (r === null || r === undefined) return;
            hx.push(1 / Math.max(1, s.players));
            hy.push(r);
          });
          const fit = theilSenFit(hx, hy);
          if (fit !== null) {
            const tierMean = tierMeanP.get(t.tier) ?? globalP;
            const n = hx.length;
            hyperParams = {
              p:
                (n * clampRate(fit.intercept) +
                  TIER_SHRINK_STRENGTH * tierMean) /
                (n + TIER_SHRINK_STRENGTH),
              c: (fit.slope * n) / (n + HYPERBOLA_SHRINK_STRENGTH),
            };
          }
        }
        // (b) counts refits on the prefix.
        let countsParams: {
          slope: number;
          intercept: number;
          nFit: LineFit;
        } | null = null;
        {
          const px: number[] = [];
          const py: number[] = [];
          kept.forEach((s, j) => {
            const r = s.rates[i];
            if (r === null || r === undefined) return;
            px.push(keptXs[j]);
            py.push(
              Math.log((clampRate(r) / 100) * s.players + EARNERS_SMOOTH),
            );
          });
          const eFit = theilSenFit(px, py);
          const nFit = theilSenFit(
            keptXs,
            kept.map((s) => Math.log(Math.max(1, s.players))),
          );
          if (eFit !== null && nFit !== null) {
            const tierMean = tierSlopeMean.get(t.tier) ?? globalSlope;
            const slope =
              (px.length * eFit.slope + TIER_SHRINK_STRENGTH * tierMean) /
              (px.length + TIER_SHRINK_STRENGTH);
            const intercept = medianOf(
              px.map((x, j) => py[j] - slope * x),
            );
            countsParams = { slope, intercept, nFit };
          }
        }
        // (c) direct logit refit on the prefix.
        let directFit: LineFit | null = null;
        {
          const qx: number[] = [];
          const qy: number[] = [];
          kept.forEach((s, j) => {
            const r = s.rates[i];
            if (r === null || r === undefined) return;
            qx.push(keptXs[j]);
            qy.push(logitOf(r));
          });
          directFit = theilSenFit(qx, qy);
        }
        for (let k = m; k < states.length; k++) {
          const actual = states[k].rates[i];
          if (actual === null || actual === undefined) continue;
          // Only the farthest predictions count: the target lies ~1.8x the
          // history span ahead, so near predictions (where the saturating
          // model looks worst and the trend models look best) must not
          // dilute the score — the hyperbola's error shrinks with distance
          // while a log-time trend's grows.
          if (k < states.length - 3) continue;
          const actualLogit = logitOf(actual);
          const preds: Array<number | null> = [
            hyperParams === null
              ? null
              : clampRate(
                  hyperParams.p +
                    hyperParams.c / Math.max(1, states[k].players),
                ),
            countsParams === null
              ? null
              : clampRate(
                  (Math.exp(
                    countsParams.intercept +
                      countsParams.slope *
                        logTimeOf(states[k].timeMs, anchorMs),
                  ) /
                    Math.max(
                      1,
                      Math.exp(
                        countsParams.nFit.intercept +
                          countsParams.nFit.slope *
                            logTimeOf(states[k].timeMs, anchorMs),
                      ),
                    )) *
                    100,
                ),
            directFit === null
              ? null
              : clampRate(
                  logisticPct(
                    directFit.intercept +
                      directFit.slope *
                        logTimeOf(states[k].timeMs, anchorMs),
                  ),
                ),
          ];
          preds.forEach((p, j) => {
            if (p !== null) errs[j].push(Math.abs(logitOf(p) - actualLogit));
          });
        }
      }
      const medErrs = errs.map((e) => (e.length > 0 ? medianOf(e) : 0));
      const exps = medErrs.map((e) => Math.exp(-e / LOO_TEMPERATURE));
      const sumExp = exps.reduce((a, b) => a + b, 0);
      weights =
        sumExp > 0 ? exps.map((e) => e / sumExp) : LOO_FALLBACK_WEIGHTS;
      // Blended LOO error: the weighted mean of the candidates' median errors,
      // the residual that sizes this trophy's spread.
      looBlendErr =
        weights[0] * medErrs[0] + weights[1] * medErrs[1] + weights[2] * medErrs[2];
    }

    let finalLogit =
      weights[0] * logitOf(candidates[0]) +
      weights[1] * logitOf(candidates[1]) +
      weights[2] * logitOf(candidates[2]);
    // Damping toward persistence in linear time (v1 idea; kept because
    // Uji 3 + Uji 4 both prove it helps — see builder-v2-notes.md).
    if (DAMPING_ENABLED) {
      const spanLin = last.timeMs - states[0].timeMs;
      const gapLin = Math.max(0, targetTimeMs - last.timeMs);
      if (spanLin > 0 && gapLin > 0) {
        const trendWeight = spanLin / (spanLin + gapLin);
        finalLogit =
          trendWeight * finalLogit + (1 - trendWeight) * logitOf(lastRate[i]);
      }
    }
    const residVar = (looBlendErr / MEDIAN_TO_SIGMA) ** 2;
    const spread = Math.min(
      SPREAD_CAP,
      Math.max(
        SPREAD_FLOOR,
        Math.sqrt(residVar + refreshVar + (SIGMA_FIT * Math.sqrt(horizonDays)) ** 2),
      ),
    );
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
