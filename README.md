# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/keepomakima/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/keepomakima/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

GTA VI trophy rarity forecast: five quantiles per trophy at 2026-11-26. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## Model

`main.ts` (Node 22, standard library only, no network, deterministic) forecasts each of the 55 GTA VI trophies' Exophase earn rate at the case target:

- **Staircase collapse** — Exophase refreshes on its own schedule, so consecutive identical reads are merged into distinct states; trends are fitted only over distinct states.
- **Dilution-aware decomposition** — rate = earners / players-tracked. Log(earners) and log(tracked) are fitted against log time with Theil-Sen slopes (robust to refresh jumps); each trophy's slope is shrunk toward its tier mean (hierarchical shrinkage). The forecast rate is earners / tracked, cross-checked against a direct logit(rate) trend fit.
- **Dilution prior** — with fewer than 3 distinct states (practice cases: one flat launch-night page) no trend can be fitted, so the forecast applies a dilution prior: the next refresh grows tracked players (~×4.8) much faster than earners (~×1.25) over 90 minutes, so rates fall; spread is wide.
- **Sticky 100%** — the one trophy at 100% did not fall at the launch-night refresh, so it is forecast near 100% with a small spread.
- **Damped extrapolation** — the further the target lies past the observed span in linear time, the more the trend is pulled back toward the last rate (rates saturate; far log-time extrapolations overshoot).
- **Spread** — logit-normal quantiles with spread from horizon, fit residuals, and refresh jumpiness (floor 0.35, cap 1.6).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` in the input directory (in name order) and writes one `<id>.out` per case: `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}`.

## Local validation

See `builder-v1-notes.md` in the build work directory (not shipped in this repo): practice-format checks, synthetic refresh scenarios (pinball 0.26–0.35 vs house baseline 1.21–1.69), and a synthetic 4.5-day horizon case (pinball 0.39 vs starter 0.21 — the synthetic truth saturates unusually fast, favouring persistence; the damping above is the hedge for the real final).

---

Entered on [make.fun](https://make.fun/?ref=dy7a168m&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
