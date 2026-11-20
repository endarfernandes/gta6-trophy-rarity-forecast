# GTA VI Trophy Rarity Forecaster

Forecasts, for each of the 55 GTA VI PS5 trophies, five quantiles (5/25/50/75/95%)
of the percentage of Exophase-tracked players who will have earned it by the
case target (2026-11-26T00:00Z for the final case). Entry for make.fun
bounty **gta6-trophy-rarity-forecast-01** ($250, deadline 21 Nov 2026 11:30 UTC).

## Run

```sh
python3 main.py            # reads $MAKEFUN_INPUT_DIR/*.in (default /input),
                           # writes $MAKEFUN_OUTPUT_DIR/*.out (default /output)
```

Python 3.13, standard library only, no network, deterministic. One case runs
in ~0.1 s (limit 8 s). Output per case:

```json
{"trophies":[{"id":1,"q":[0.01,0.01,0.025,0.0664,0.2701]}, ...]}
```

55 rows, ids 1–55 once each, quantiles nondecreasing, each 0–100.

## Method

Per trophy, three median models are fit in log-time `x = ln(1 + hours since
release 2026-11-19T00:00Z)`, in logit space:

- **trend** — recency-weighted (24 h half-life) least squares of logit(rate)
  on x, slope shrunk toward the trophy's class-mean slope.
- **logistic** — saturation curve `rate = A / (1 + exp(-(a + b x)))` with a
  research-based asymptote A per trophy class, fit twice (A and 0.6·A) and
  averaged. Early launch growth is steeper in log-time than a curve aimed at
  the eventual rate, so the 0.6·A fit carries the launch shape.
- **counts** — log(earners) and log(players tracked) fit separately on x,
  then rate = earners / tracked. The only component that models the
  denominator (dilution) explicitly.

Medians are combined with weights from an internal backtest: each model is
refit on data truncated 12 h and 24 h before the checkpoint and scored
against the realised last read. Errors are averaged (the competition loss is
a mean over trophies) and weights are computed globally, then blended with
fixed priors (logistic .50, trend .35, counts .15).

**Class prior slope.** A research-based prior day-7 rate per class (see
`PRIOR7` in main.py) supplies a slope prior, anchored to day 7 (168 h) so
the implied slope does not depend on the case horizon. When a trophy's
history is flat — e.g. printed at the 0.01% resolution floor for two days,
below Exophase's 2-decimal print resolution — the fits follow the prior
slope (capped at 3.0 logit per unit log-time); when the history moves, the
data slope dominates (prior weight 1.5 against the number of distinct
reads). Single-point fits are anchored at the checkpoint, not at the
observation's own time.

**Spread.** Logit sigma = sqrt(base² + disagreement² + (backtest error ×
horizon scale)²), base = 0.45·√(horizon days), floored 0.35, capped 2.5.
A trophy still at the 0.01% print floor whose class prior is far above it
gets a wide band (spread ≥ 1.8): the history carries no signal about its
takeoff, so the band states that ignorance instead of the (silent)
backtest error.

**Special cases.** Flat history with checkpoint ≤ 48 h after release (a
stale page — a refresh jump is coming, tracked players jump, rates fall):
median shifted down by ln(0.4), spread ≥ 1.1. Trophy 2 ("Just Like
Paradise", ~100% in every read): forecast to stay at 99.99% with a narrow
band. Rare trophies at the 0.01% print floor get a small median floor
(0.025%) so the forecast never claims a rate below what the page can print.

## Validation (summary — full report in the project workspace)

- Practice cases p00/p01: valid output, correct schema, ~0.1 s per case.
- Synthetic final-like cases (cohort simulator, two ground-truth families —
  lognormal and Weibull time-to-earn — × 7 scenarios × 2 seeds = 28 cases),
  scored with the official pinball loss on the logit scale:
  - lognormal family: **0.76** mean (worst 0.90) vs starter baseline 1.75,
    house trend clone 1.17.
  - Weibull family: **0.28** mean (worst 0.32) vs starter 0.66, trend clone 0.37.
  - v1 (superseded) scored 0.94 / 0.32; the v2 changes (prior-slope cap
    3.0, day-7 anchoring, checkpoint anchor, floor spread ≥ 1.8, global
    weights only, trophy 2 at 99.99) improved all 28 cases with zero
    regressions — see `validation_report.md` and the independent review in
    `scratch/exp/INDEPENDENT_REVIEW.md`.
- Ablation (4 configs, same seeds): the shipped config (prior strength 1.5,
  spread factor 1.0) has the best combined mean across both families; the
  prior slope and the backtest-error spread term were each tested on and off.
- Determinism: two runs on the same inputs are byte-identical.

## Research grounding

`research_analogues.md` in the project workspace (every number sourced, URL +
access date 2026-11-20): GTA V eventual funnel (Steam first mission 81.9%,
final mission 23.8%, 100% 6.0%), PSN-wide vs tracked denominator gap
(50–180×), RDR2 / Cyberpunk / Elden Ring eventual rates, the only two dated
early-window curves (God of War Ragnarök ~25% story at 14 days; Phantom
Blade Zero final 0% day 2 → 21.6% day 11), GTA VI story ~50–60 h and
platinum ~200 h, and Exophase's definition (earners ÷ tracked players who
earned ≥ 1 trophy). No GTA V week-1 time series exists — that gap is stated
explicitly, and it is why the model leans on the case's own history plus
backtested weights rather than on any single analogue.

## Files

- `main.py` — the forecaster (the graded entry).
- `README.md` — this file.
- Workspace (not part of the submission): `research_analogues.md`,
  `research_report.md`, `source_registry.csv`, `trophy_classification.csv`,
  `forecast_55_trophies.csv` (illustrative forecast on a synthetic final-like
  case), `validation_report.md`, `sensitivity_analysis.md`, `scratch/`
  (simulator + test harness).
