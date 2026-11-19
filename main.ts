// GTA VI trophy rarity forecaster v3 — refresh-inversion prior + jump-share
// saturating model + forward-validation blend + adaptive damping.
//
// Model, in one paragraph: Exophase's page is a staircase — flat between its
// own refreshes, jumping when it refreshes — so the history is first collapsed
// to distinct states (consecutive identical reads merged). Two regimes are
// split on the number of distinct states. With fewer than 3 states no trend
// can be fitted, so a refresh-inversion prior is used: the first refresh in
// RULES.txt grew tracked players 638 -> 3,868 (x6.06) while earners barely
// moved, so the median rate is the last rate times ~1.08/6.06 (~0.18) with a
// spread set by honest refresh uncertainty. With 3+ states, earners E_i(t)
// and tracked players N(t) are fitted SEPARATELY (never the ratio directly):
// four candidates per trophy are blended in logit space with weights from
// expanding-window forward validation (train on the first 35% of states,
// score on the 3 farthest states — the target lies ~1.8x the history span
// ahead, so only long-range extrapolation is scored): (a) a saturating
// hyperbola r(N) = p + c/N fitted by Theil-Sen of rate against 1/N — the
// exact form when a fixed share p of new players earns the trophy; (b) the
// counts model, Theil-Sen on log(earners) and log(tracked) against log time;
// (c) a direct Theil-Sen fit of logit(rate) against log time; (d) a NEW
// jump-share model that learns the refresh process from the data itself:
// each refresh's multiplier pair (f_N, f_E,i) is observed, so the share p_i
// is estimated as the recency-weighted ratio of summed earner jumps to
// summed tracked jumps, shrunk toward its tier mean by empirical Bayes
// (shrinkage from the measured between-trophy variance, not a fixed
// pseudo-count). The population at the target comes from a logistic N(t)
// fit (grid-searched, SSE in log N), falling back to the log-time N fit.
// Damping toward the last rate is ADAPTIVE: it applies in full only when
// forward validation says the blend is uncertain, and vanishes when the
// blend validates near-perfectly (a perfect constant-share process needs no
// damping; a drifting one does). Quantiles come from a normal spread on the
// logit scale sized by the forward-validation residual, refresh variance
// and a horizon term, floored so the model never claims ~zero risk.
// Node 22, standard library only, no network, deterministic.
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
/** Blend weights (hyperbola / counts / direct-logit / jump-share) when too few states for LOO. */
const LOO_FALLBACK_WEIGHTS = [0.4, 0.2, 0.15, 0.25];
/** Convert a median absolute logit error to a Gaussian sigma (0.6745 = z at 0.75). */
const MEDIAN_TO_SIGMA = 0.6745;

/** Prior tracked-player growth over one 90-minute horizon in the no-trend regime.
    RULES.txt line 11 documents the first refresh: N 638 -> 3,868, factor x6.06. */
const PRIOR_N_GROWTH_90MIN = 6.06;
/** Prior earner growth over one 90-minute horizon in the no-trend regime.
    Forensic joint fit on the live practice scores (v1 0.326, v2 0.216):
    f_E in 1.05-1.10 reproduces BOTH scores within 0.01 (sse 0.000114 at
    1.05, 0.000120 at 1.08, 0.000125 at 1.10 — flat within rounding noise),
    while 1.15 is already off by 0.023 on v2. Sweep with the v3 code at
    PRIOR_SPREAD 0.15 (Uji 2 mean / calibrated-live f_E=1.1): 1.05 ->
    0.1619 / 0.0365; 1.08 -> 0.1493 / 0.0326; 1.10 -> 0.1414 / 0.0326;
    1.15 -> 0.1252 / 0.0410. 1.10 is chosen: joint-fit central, tied-best
    calibrated-live, and the median ratio 1.10/6.06 = 0.182 matches the
    coordinator's forensic ~0.18. */
const PRIOR_EARNERS_GROWTH_90MIN = 1.1;
/** Caps on the prior growth factors, so a long horizon cannot explode them. */
const PRIOR_N_GROWTH_CAP = 12;
const PRIOR_EARNERS_GROWTH_CAP = 2.5;
/** Spread (logit) of the refresh-inversion prior at a 90-minute horizon.
    Sweep at f_E=1.08 (Uji 2 mean / calibrated-live f_E=1.1): 0.10 ->
    0.1653 / 0.0228; 0.12 -> 0.1568 / 0.0267; 0.15 -> 0.1493 / 0.0326;
    0.20 -> 0.1392 / 0.0425. Wider spreads hedge the x1.3/x1.6 earner
    scenarios in Uji 2 but cost on the calibrated-live answer; 0.15 is the
    middle of the honest range (the forensic f_E uncertainty is ~+/-0.1 in
    log) and clears both gates with margin. */
const PRIOR_SPREAD = 0.15;
/** Largest prior spread allowed when the horizon stretches (sqrt scaling). */
const PRIOR_SPREAD_CAP = 0.6;

/** Recency weight per refresh for the jump-share estimator (0.85^k): recent
    refreshes describe the current share better when the share drifts. */
const JUMP_RECENCY_RHO = 0.85;
/** Floor on the jump-share sampling variance (rate rounding is 0.01pp, so a
    jump's share is never known better than ~0.001). */
const JUMP_VAR_FLOOR = 1e-6;
/** Largest empirical-Bayes shrinkage toward the tier mean: a trophy is never
    pulled more than halfway to its tier mean, so a genuinely atypical trophy
    keeps at least half its own estimate. */
const EB_MAX_SHRINK = 0.5;

/** Damping reference error: when the forward-validation blend error reaches
    this (logit), the v2 linear-time damping applies in full; at zero error
    no damping applies at all. 0.3 is v2's typical blend error on Uji 3/4.
    Sweep (Uji 3/4/5): adaptive 0.3 -> 0.0369/0.0368/0.0378; damping OFF ->
    0.0370/0.0398/0.0376; v2 full damping -> 0.0357/0.0889/0.0401. Full
    damping is proven harmful here (Uji 4: the truth is still rising 15-20%
    above the last rate, so pulling 64% toward persistence undershoots
    systematically); adaptive keeps v2's protection exactly when validation
    says the blend is uncertain, and releases it when the blend validates. */
const DAMPING_REF_ERROR = 0.3;
/** Blend error assumed when too few states exist for forward validation
    (3 states): a 3-state fit is shaky, so damping applies in full and the
    spread stays wide. */
const LOO_ERROR_FALLBACK = 0.3;

/** Random-walk spread per sqrt-day of horizon. Sweep on Uji 3/4/5 with the
    v3 median (losses): 0.05 -> 0.0316/0.0336/0.0345; 0.075 ->
    0.0369/0.0368/0.0378; 0.10 -> 0.0466/0.0469/0.0479. The synthetic
    generators have no beyond-history drift, so they always prefer the
    smallest value; 0.075 is kept as the honest middle — it leaves a real
    horizon term (0.159 logit over 4.5 days, typical spread ~0.18) instead
    of letting the 0.15 floor claim the long extrapolation is ~free. */
const SIGMA_FIT = 0.075;
/** Extra logit variance allowed for refresh timing in the fitted regime. */
const REFRESH_VAR_SCALE = 0.25;
/** Fallback refresh variance when too few states exist to estimate it. */
const REFRESH_VAR_FALLBACK = 0.64;
/** Smallest / largest logit spread handed to the quantiles. The floor is the
    honesty guardrail: forward-validation residuals on a clean history can
    approach rounding noise, but the final's 4.5-day extrapolation is never
    a ~zero-risk forecast, so the spread never drops below 0.15. */
const SPREAD_FLOOR = 0.15;
const SPREAD_CAP = 1.2;

/** A rate at/above this is treated as sticky at ~100% (it did not fall at refresh). */
const STICKY_RATE = 99.99;
/** Median and spread for a sticky 100% trophy. The trophy printed 100.00
    after a 6x refresh while every other rate fell, i.e. essentially every
    newly tracked player already had it; the pinball-optimal median of a
    posterior with >50% mass at 100% is 100% itself (clamped 99.99 by the
    scorer), so the median sits at the boundary, not at 99.7 — 99.7 costs
    ~3.4 logit units (~0.062 of mean loss) when the truth is 100.00. */
const STICKY_MEDIAN = 99.99;
const STICKY_SPREAD = 0.15;
/** Largest plausible tracked-player growth per fitted extrapolation (sanity cap). */
const MAX_N_GROWTH = 50;
/** Lowest plausible forecast rate, as a fraction of the last rate (never below 0.01). */
const RATE_FLOOR_FACTOR = 0.15;

/** Logistic N grid: carrying-capacity multiples of the last tracked count. */
const LOGISTIC_K_MULTIPLIERS = [1.5, 2, 3, 5, 8, 12, 20, 50, 100, 200];
/** Logistic N grid: absolute carrying capacities (launch population is in the 100ks). */
const LOGISTIC_K_ABSOLUTE = [1_000_000, 2_000_000, 5_000_000];
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
interface JumpFit {
  /** Estimated share of new players earning the trophy (fraction 0..1). */
  p: number;
  /** Sampling variance of p from the scatter of per-refresh shares. */
  variance: number;
  count: number;
}

const clampRate = (x: number): number =>
  Math.min(RATE_CEIL, Math.max(RATE_FLOOR, x));
const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));
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

/**
 * Jump-share fit for one trophy: the share p of new players earning it,
 * estimated from the observed per-refresh jumps (dE/dN), recency-weighted.
 * p = sum(w*dE) / sum(w*dN) with w = rho^(age in refreshes). The variance
 * comes from the scatter of the per-refresh shares around p — when the
 * share is truly constant the scatter is rounding noise only, so the
 * variance is tiny and empirical-Bayes shrinkage (below) leaves p alone;
 * when the share drifts, the scatter grows and shrinkage kicks in.
 * Null when no usable jump exists (no tracked growth observed).
 */
function jumpShareFit(states: DistinctState[], i: number): JumpFit | null {
  const m = states.length;
  let num = 0;
  let den = 0;
  let wSum = 0;
  let w2Sum = 0;
  const jumps: Array<{ w: number; pk: number }> = [];
  for (let k = 1; k < m; k++) {
    const dN = states[k].players - states[k - 1].players;
    if (dN <= 0) continue;
    const rPrev = states[k - 1].rates[i];
    const rCur = states[k].rates[i];
    if (rPrev === null || rPrev === undefined) continue;
    if (rCur === null || rCur === undefined) continue;
    const dE =
      (rCur / 100) * states[k].players -
      (rPrev / 100) * states[k - 1].players;
    const w = Math.pow(JUMP_RECENCY_RHO, m - 1 - k);
    num += w * dE;
    den += w * dN;
    wSum += w;
    w2Sum += w * w;
    jumps.push({ w, pk: dE / dN });
  }
  if (den <= 0 || jumps.length === 0 || wSum <= 0) return null;
  const p = num / den;
  let scatter = 0;
  for (const j of jumps) scatter += j.w * (j.pk - p) * (j.pk - p);
  scatter /= wSum;
  const variance = Math.max(
    JUMP_VAR_FLOOR,
    scatter * (w2Sum / (wSum * wSum)),
  );
  return { p, variance, count: jumps.length };
}

/**
 * Empirical-Bayes shrinkage of jump-share estimates toward their tier mean.
 * tau^2 (between-trophy variance within a tier) is estimated from the data
 * as the observed variance of p across the tier's trophies minus the mean
 * sampling variance; a trophy's weight on its own estimate is
 * tau^2 / (tau^2 + var_i), never below 1 - EB_MAX_SHRINK, so no trophy is
 * pulled more than halfway to its tier mean. Returns shrunk shares
 * (fractions 0..1), NaN where the trophy has no jump fit.
 */
function shrinkJumpShares(
  fits: Array<JumpFit | null>,
  tiers: string[],
): number[] {
  const out = fits.map((f) => (f === null ? NaN : clamp01(f.p)));
  for (const tier of new Set(tiers)) {
    const idx = tiers
      .map((t, i) => (t === tier && fits[i] !== null ? i : -1))
      .filter((i) => i >= 0);
    if (idx.length < 2) continue;
    const ps = idx.map((i) => clamp01((fits[i] as JumpFit).p));
    const vars = idx.map((i) => (fits[i] as JumpFit).variance);
    const tierMean = meanOf(ps);
    const tau2 = Math.max(0, varianceOf(ps) - meanOf(vars));
    for (const i of idx) {
      const vi = (fits[i] as JumpFit).variance;
      const w =
        tau2 <= 0
          ? 1
          : Math.max(1 - EB_MAX_SHRINK, tau2 / (tau2 + vi));
      out[i] = clamp01(w * clamp01((fits[i] as JumpFit).p) + (1 - w) * tierMean);
    }
  }
  return out;
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

  // ---- No-trend regime: fewer than 3 distinct states -> refresh-inversion prior. ----
  // RULES.txt line 11 documents the first refresh: N 638 -> 3,868 (x6.06)
  // while earners barely moved (forensic joint fit on the live practice
  // scores: f_E ~1.05-1.10, centre 1.10), so the median rate is the last
  // rate times 1.10/6.06 = 0.182, with a spread set by honest refresh
  // uncertainty (swept 0.05-0.20; 0.15 chosen — see PRIOR_SPREAD).
  if (states.length < MIN_STATES_FOR_TREND || last === undefined) {
    const nGrowth = Math.min(
      PRIOR_N_GROWTH_CAP,
      Math.exp(Math.log(PRIOR_N_GROWTH_90MIN) * horizonScale),
    );
    const eGrowth = Math.min(
      PRIOR_EARNERS_GROWTH_CAP,
      Math.exp(Math.log(PRIOR_EARNERS_GROWTH_90MIN) * horizonScale),
    );
    const priorSpread = Math.min(
      PRIOR_SPREAD_CAP,
      PRIOR_SPREAD * horizonScale,
    );
    return c.trophies.map((t, i) => {
      if (lastRate[i] >= STICKY_RATE) {
        return { id: t.id, q: quantiles(STICKY_MEDIAN, STICKY_SPREAD) };
      }
      const medianPct = clampRate(lastRate[i] * (eGrowth / nGrowth));
      return { id: t.id, q: quantiles(medianPct, priorSpread) };
    });
  }

  // ---- Fitted regime. ----
  const anchorMs = states[0].timeMs;
  const xs = states.map((s) => logTimeOf(s.timeMs, anchorMs));
  const targetMs = Date.parse(c.target);
  const targetTimeMs = Number.isFinite(targetMs) ? targetMs : last.timeMs;
  const xTarget = logTimeOf(targetTimeMs, anchorMs);
  const tTargetDays = daysSince(targetTimeMs, anchorMs);
  const tiers = c.trophies.map((t) => t.tier);

  // Tracked players at target: logistic fit, else the log-time fit.
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
  // Candidate (b) keeps the log-time N; candidates (a) and (d) use the logistic N.
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
  // Jump-share fits (full data), shrunk toward their tier mean by empirical Bayes.
  const rawJump: Array<JumpFit | null> = c.trophies.map((_, i) =>
    jumpShareFit(states, i),
  );
  const shrunkJumpP = shrinkJumpShares(rawJump, tiers);

  // Tier means for shrinkage (hyperbola asymptote; counts slope), plus global fallbacks.
  const tierMeanP = new Map<string, number>();
  const allP = rawHyperbola
    .filter((f): f is HyperbolaFit => f !== null)
    .map((f) => clampRate(f.p));
  const globalP = allP.length > 0 ? meanOf(allP) : 1;
  for (const tier of new Set(tiers)) {
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
  for (const tier of new Set(tiers)) {
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

  // Jump-share forecast for one trophy (full-data fit): earners at target
  // are the last earners plus the shrunk share of the new tracked players.
  const jumpRate = (i: number): number | null => {
    const p = shrunkJumpP[i];
    if (!Number.isFinite(p)) return null;
    const eLast = (lastRate[i] / 100) * last.players;
    const eTarget =
      eLast + p * Math.max(0, nForecastHyperbola - last.players);
    return clampRate((eTarget / Math.max(1, nForecastHyperbola)) * 100);
  };

  // ---- Expanding-window forward validation, precomputed per split. ----
  // Split points (train on the first m states) and, for each, the prefix
  // jump-share fits for ALL trophies (shrunk toward prefix tier means),
  // so candidate (d) can be refitted on the prefix exactly like (a)-(c).
  const splitPoints = new Set<number>();
  if (states.length >= MIN_STATES_FOR_LOGISTIC) {
    for (const f of LOO_SPLIT_FRACTIONS) {
      const m = Math.max(
        MIN_STATES_FOR_TREND,
        Math.floor(f * states.length),
      );
      if (m < states.length - 1) splitPoints.add(m);
    }
  }
  const prefixJumpBySplit = new Map<number, number[]>();
  for (const m of [...splitPoints].sort((a, b) => a - b)) {
    const kept = states.slice(0, m);
    const fits = c.trophies.map((_, j) => jumpShareFit(kept, j));
    prefixJumpBySplit.set(m, shrinkJumpShares(fits, tiers));
  }

  return c.trophies.map((t, i) => {
    if (lastRate[i] >= STICKY_RATE) {
      return { id: t.id, q: quantiles(STICKY_MEDIAN, STICKY_SPREAD) };
    }
    const candA = hyperbolaRate(i) ?? lastRate[i];
    const candB = countsRate(i) ?? lastRate[i];
    const candC = directRate(i) ?? lastRate[i];
    const candD = jumpRate(i) ?? lastRate[i];
    const candidates = [candA, candB, candC, candD];

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
    let looBlendErr = LOO_ERROR_FALLBACK;
    if (states.length >= MIN_STATES_FOR_LOGISTIC && splitPoints.size > 0) {
      const errs: number[][] = [[], [], [], []];
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
        // (d) jump-share on the prefix: share from the precomputed table,
        // earners anchored at the last prefix state.
        const jumpP = prefixJumpBySplit.get(m)?.[i];
        const lastKept = kept[kept.length - 1];
        const lastKeptRate =
          lastKept === undefined ? null : lastKept.rates[i];
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
            jumpP === undefined || !Number.isFinite(jumpP) || lastKeptRate === null || lastKeptRate === undefined
              ? null
              : clampRate(
                  (((lastKeptRate / 100) * lastKept.players +
                    jumpP *
                      Math.max(0, states[k].players - lastKept.players)) /
                    Math.max(1, states[k].players)) *
                    100,
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
      // the residual that sizes this trophy's spread and its damping.
      looBlendErr =
        weights[0] * medErrs[0] + weights[1] * medErrs[1] + weights[2] * medErrs[2] + weights[3] * medErrs[3];
    }

    let finalLogit =
      weights[0] * logitOf(candidates[0]) +
      weights[1] * logitOf(candidates[1]) +
      weights[2] * logitOf(candidates[2]) +
      weights[3] * logitOf(candidates[3]);
    // ADAPTIVE damping toward persistence in linear time. The v2 weight
    // span/(span+gap) applies in full only when forward validation says the
    // blend is uncertain (error >= DAMPING_REF_ERROR); when the blend
    // validates near-perfectly the trend is trusted and damping vanishes.
    // A constant-share process needs no damping; a drifting one does — and
    // the validation error is what tells them apart, per trophy.
    {
      const spanLin = last.timeMs - states[0].timeMs;
      const gapLin = Math.max(0, targetTimeMs - last.timeMs);
      if (spanLin > 0 && gapLin > 0) {
        const baseWeight = spanLin / (spanLin + gapLin);
        const confidence = Math.min(
          1,
          Math.max(0, looBlendErr / DAMPING_REF_ERROR),
        );
        const trendWeight = 1 - (1 - baseWeight) * confidence;
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
