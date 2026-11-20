#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster — make.fun gta6-trophy-rarity-forecast-01.

Reads one case (/input/<id>.in): every Exophase read of the GTA VI PS5 trophy
page before the case checkpoint, and forecasts, for each of the 55 trophies,
five quantiles (5/25/50/75/95%) of its earn rate at the case target
(2026-11-26T00:00Z for the final case), scored by pinball loss on the logit
scale.

Method (stdlib only, deterministic, no network):
  * Per trophy, three median models are fit in log-time x = ln(1 + hours
    since release 2026-11-19T00:00Z):
      - trend:     weighted least squares of logit(rate) on x (recency weights,
                   slope shrunk toward the trophy's class mean slope).
      - logistic:  saturation curve rate = A / (1 + exp(-(a + b x))) with a
                   research-based asymptote A per trophy class; fit twice
                   (A and 0.6*A) and averaged, because early launch growth is
                   steeper in log-time than a curve aimed at the eventual rate.
      - counts:    log(earners) and log(players tracked) fit separately on x,
                   then rate = earners / tracked. This is the only component
                   that models the denominator (dilution) explicitly.
  * The three medians are combined in logit space with weights from an
    internal backtest: each model is refit on data truncated 12h and 24h
    before the checkpoint and scored against the realised last read.
    Errors are averaged (the competition loss is a mean over trophies),
    and weights are computed globally, then blended with fixed priors
    (logistic .50, trend .35, counts .15).
  * A class prior day-7 rate (from the research analogues) supplies a
    slope prior: when a trophy's history is flat — e.g. printed at the
    0.01% resolution floor for two days — the fits follow the prior
    slope; when the history moves, the data slope dominates. This is
    the only place the research rates enter the forecast directly.
  * Spread (logit sigma) combines a random-walk term that grows with the
    horizon, the disagreement between the three medians, and the backtest
    error scaled to the forecast horizon. It is floored and capped so the
    5%-95% band stays honest: wide enough to cover refresh jumps and model
    error, narrow enough not to pay at every trophy.
  * A trophy still at the 0.01% print floor whose class prior is far above
    it gets a wide band (spread >= 1.8): the history carries no signal
    about its takeoff, so the band states that ignorance instead of the
    (silent) backtest error.
  * A flat history (every read identical — Exophase had not refreshed the
    page) means a refresh jump is coming: medians are shifted down (tracked
    players jump, rates fall) and the spread is widened. This is what the
    practice cases test; the final case's history is not flat, so it does
    not trigger there.
  * Trophy 2 ("Just Like Paradise") sits at ~100% in every read; it is
    forecast to stay at 99.99% with a narrow band. Rare trophies whose last rate
    is at the 0.01% print floor get a small median floor so the forecast
    does not claim a rate below what the page can print.

Research grounding (see research_analogues.md in the project workspace):
GTA V's eventual funnel (Steam: first mission 81.9%, final mission 23.8%,
100% 6.0%), RDR2 / Cyberpunk / Elden Ring eventual rates, the only two
dated early-window curves (God of War Ragnarok ~25% story at 14 days;
Phantom Blade Zero final 0% day 2 -> 21.6% day 11), GTA VI story ~50-60h
and platinum ~200h, and Exophase's definition (earners / tracked players
who earned >= 1 trophy). No statistic is invented: every asymptote below
is a prior that the data-driven fits can and do override, and the backtest
weights let the case's own history choose between the models.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Prior day-7 rates (%) per trophy class, from the research analogues
# (GTA V eventual funnel, the two dated early-window curves, GTA VI's
# 50-60h story / ~200h platinum). Used ONLY as a slope prior when a
# trophy's own history is flat or nearly so — e.g. a trophy printed at
# 0.01% for two days, below Exophase's 2-decimal resolution, whose rise
# the history cannot show. When the history moves, the data slope
# dominates (the prior gets weight 1.5 against the number of distinct
# reads). PRIOR_STRENGTH is that prior weight.
PRIOR7 = {"story1": 45.0, "story2": 28.0, "story3": 18.0, "story4": 12.0,
          "storyfinal": 5.0, "strand": 6.0, "activity": 15.0,
          "combat": 5.0, "collect": 1.5, "grind": 0.5,
          "platinum": 0.05, "completion": 0.1, "first": 99.0,
          "silver_misc": 3.0, "gold_misc": 10.0, "bronze_misc": 4.0}
PRIOR_STRENGTH = 1.5
# SPREAD_BT_FACTOR scales the backtest-error term in the spread
# (0.5 = original); SPREAD_CAP is the maximum logit sigma.
SPREAD_BT_FACTOR = 1.0
SPREAD_CAP = 2.5
# FLOOR_SPREAD is the minimum logit sigma for a trophy still at the
# 0.01% print floor whose class prior is far above it (see forecast_case).
FLOOR_SPREAD = 1.8

# Prior weights for the three median models (blended with backtest weights).
PRIOR_W = {"logistic": 0.50, "trend": 0.35, "counts": 0.15}


def _log_odds_frac(r):
    r = min(max(r, 1e-6), 0.999999)
    return math.log(r / (1.0 - r))


def prior_slope(cls, last, x_at, x_target, asymp=None):
    """Slope in log-odds space that would take `last` to the class prior
    day-7 rate by the target. In trend space the log-odds are logit(rate
    in %); in logistic space they are log((rate/A) / (1 - rate/A)).
    Floored at 0 (the prior never predicts a decline on its own) and
    capped to avoid explosive priors."""
    p7 = PRIOR7.get(cls, 5.0)
    if asymp is not None:
        return min(max((_log_odds_frac(min(p7 / asymp, 0.99)) -
                        _log_odds_frac(min(last / asymp, 0.99))) /
                       max(x_target - x_at, 0.1), 0.0), 3.0)
    return min(max((logit(min(p7, 99.0)) - logit(last)) /
                   max(x_target - x_at, 0.1), 0.0), 3.0)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    lg = max(-30.0, min(30.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (as JavaScript Math.round does it)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    """ISO-8601 to aware datetime; returns None when unparsable."""
    if not isinstance(s, str):
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def hours_since_release(dt):
    if dt is None:
        return None
    return (dt - RELEASE).total_seconds() / 3600.0


# ---------------------------------------------------------------- classification
def classify(trophy):
    """(class label, saturation asymptote A in %) from name/description/tier.

    A is the level the logistic curve heads for: roughly the Exophase-tracked
    rate the trophy approaches over the first month, from the research
    analogues (GTA V / RDR2 / Cyberpunk eventual funnels, the two dated
    early-window curves, GTA VI's 50-60h story and ~200h platinum). It is a
    prior only: the fits below are driven by the case history, and the
    backtest weights can discard the logistic component entirely.
    """
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    d = name + " " + desc
    if tier == "platinum":
        return ("platinum", 0.9)
    if "100% completion" in d:
        return ("completion", 2.0)
    if "chapter 1" in d:
        return ("story1", 60.0)
    if "chapter 2" in d:
        return ("story2", 45.0)
    if "chapter 3" in d:
        return ("story3", 34.0)
    if "chapter 4" in d:
        return ("story4", 27.0)
    if "final mission for" in d:
        return ("strand", 12.0)
    if "final mission" in d:
        return ("storyfinal", 22.0)
    if "gone fishing" in d:
        return ("first", 99.0)
    if "250 stunts" in d:
        return ("grind", 2.5)
    if "shooting range" in d:
        return ("grind", 3.5)
    if "drone race" in d:
        return ("grind", 3.0)
    if "20 different races" in d:
        return ("grind", 5.0)
    if "8 mission strands" in d:
        return ("grind", 5.0)
    if "meditate" in d:
        return ("collect", 2.5)
    if "collectible card" in d:
        return ("collect", 2.0)
    if "waink" in d:
        return ("collect", 1.5)
    if "50 challenges" in d:
        return ("grind", 2.5)
    if "explore all of leonida" in d:
        return ("collect", 4.0)
    if "laundering" in d or "launder" in d:
        return ("grind", 3.0)
    if "spend $100,000" in d:
        return ("grind", 7.0)
    if "garage properties" in d:
        return ("collect", 4.0)
    if "register 10 vehicles" in d:
        return ("grind", 7.0)
    if "discover 100 vehicles" in d:
        return ("grind", 10.0)
    if "each type of activity" in d:
        return ("activity", 25.0)
    if "arcade game" in d:
        return ("activity", 20.0)
    if "fishing challenge" in d:
        return ("activity", 28.0)
    if "bank vault" in d or "illegal operation" in d:
        return ("combat", 6.0)
    if any(k in d for k in ("complete breakfast run", "complete the cartel",
                            "complete child's play", "complete headliners",
                            "complete vcia", "complete operation")):
        return ("strand", 14.0)
    if any(k in d for k in ("rob ", "robbery", "wanted level", "combat focus",
                            "vcia", "rhino", "jewelry", "armored truck")):
        return ("combat", 12.0)
    if tier == "silver":
        return ("silver_misc", 5.0)
    if tier == "gold":
        return ("gold_misc", 15.0)
    return ("bronze_misc", 8.0)


# ---------------------------------------------------------------- fitting
def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns (a, b, resid_sd)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return (ys[0] if ys else 0.0, 0.0, 0.0)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        resid = [y - my for y in ys]
        var = sum(r * r for r in resid) / max(len(ys) - 1, 1)
        return (my, 0.0, math.sqrt(var))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(len(ys) - 2, 1)
    return (a, b, math.sqrt(var))


def dedup_reads(history):
    """Collapse consecutive identical reads (Exophase refresh steps).

    Returns a list of dicts {t (hours since release), n (players tracked),
    rates (list, forward-filled per trophy)}. Reads with no usable rates
    are dropped.
    """
    rows = []
    for h in history:
        if not isinstance(h, dict):
            continue
        rates = h.get("rates")
        if not isinstance(rates, list) or not rates:
            continue
        t = hours_since_release(parse_time(h.get("slot") or h.get("at")))
        if t is None:
            continue
        n = h.get("players_tracked")
        try:
            n = float(n)
        except (TypeError, ValueError):
            n = None
        rows.append({"t": t, "n": n, "rates": list(rates)})
    rows.sort(key=lambda r: r["t"])
    # Forward-fill null rates per trophy, then drop rows still all-null.
    filled = []
    last = None
    for r in rows:
        if last is None:
            cur = [v if v is not None else None for v in r["rates"]]
        else:
            cur = [v if v is not None else last[i]
                   for i, v in enumerate(r["rates"])]
        if all(v is None for v in cur):
            continue
        last = cur
        filled.append({"t": r["t"], "n": r["n"], "rates": cur})
    # Drop consecutive duplicates (same rates and same tracked count).
    out = []
    for r in filled:
        if out and out[-1]["rates"] == r["rates"] and out[-1]["n"] == r["n"]:
            continue
        out.append(r)
    return out


def recency_weights(times, at_t, half_life_h=24.0):
    return [0.5 ** (max(at_t - t, 0.0) / half_life_h) for t in times]


def fit_trend(times, vals, at_t, x_target, shrink_b=None, prior_slope=None):
    """logit(rate) ~ a + b * ln(1+t); returns (median_logit, resid_sd).
    The slope is shrunk toward the class-mean slope, then blended with
    the class prior slope (weight PRIOR_STRENGTH against the number of
    distinct reads): a flat history follows the prior, a moving history
    follows the data. With a single point the line is drawn through it
    with the prior slope."""
    pts = [(t, v) for t, v in zip(times, vals) if v is not None]
    if not pts:
        return (logit(0.05), 0.5)
    if len(pts) == 1:
        if prior_slope is not None:
            xt = math.log1p(max(x_target, 0.0))
            # Anchor at the checkpoint, not at the observation's own time:
            # a single distinct value means the rate persisted unchanged
            # to the checkpoint, so the slope applies from there.
            xl = math.log1p(max(at_t, 0.0))
            return (logit(pts[0][1]) + prior_slope * (xt - xl), 0.5)
        return (logit(pts[0][1]), 0.5)
    xs = [math.log1p(max(t, 0.0)) for t, _ in pts]
    ys = [logit(v) for _, v in pts]
    ws = recency_weights([t for t, _ in pts], at_t)
    a, b, sd = wls(xs, ys, ws)
    if shrink_b is not None:
        b = 0.75 * b + 0.25 * shrink_b
    if prior_slope is not None:
        b = (len(pts) * b + PRIOR_STRENGTH * prior_slope) / (len(pts) + PRIOR_STRENGTH)
    a = sum(w * (y - b * x) for w, x, y in zip(ws, xs, ys)) / sum(ws)
    return (a + b * math.log1p(max(x_target, 0.0)), sd)


def fit_logistic(times, vals, at_t, x_target, asymp, prior_slope=None):
    """rate = A / (1 + exp(-(a + b x))); returns median_logit or None.
    With fewer than 3 usable points the line is drawn through the last
    point with the prior slope (when one is given); otherwise None. The
    fitted slope is floored at 0 and blended with the prior slope."""
    pts = []
    for t, v in zip(times, vals):
        if v is None:
            continue
        v = min(max(v, 0.005), asymp * 0.995)
        if v <= 0.005 or v >= asymp * 0.995:
            continue
        pts.append((t, v))
    if len(pts) < 3:
        if prior_slope is None or not pts:
            return None
        t_last, v_last = pts[-1]
        a = math.log(v_last / (asymp - v_last)) - prior_slope * math.log1p(max(at_t, 0.0))
        lg = a + prior_slope * math.log1p(max(x_target, 0.0))
        lg = max(-30.0, min(30.0, lg))
        r = asymp / (1.0 + math.exp(-lg))
        return logit(r)
    xs = [math.log1p(max(t, 0.0)) for t, _ in pts]
    ys = [math.log(v / (asymp - v)) for _, v in pts]
    ws = recency_weights([t for t, _ in pts], at_t)
    a, b, sd = wls(xs, ys, ws)
    b = max(b, 0.0)
    if prior_slope is not None:
        b = (len(pts) * b + PRIOR_STRENGTH * prior_slope) / (len(pts) + PRIOR_STRENGTH)
        a = sum(w * (y - b * x) for w, x, y in zip(ws, xs, ys)) / sum(ws)
    lg = a + b * math.log1p(max(x_target, 0.0))
    lg = max(-30.0, min(30.0, lg))
    r = asymp / (1.0 + math.exp(-lg))
    return logit(r)


def fit_counts(times, vals, ns, at_t, x_target):
    """log(earners) and log(tracked) on x; returns median_logit or None."""
    pts = [(t, v, n) for t, v, n in zip(times, vals, ns)
           if v is not None and n and n > 0]
    if len(pts) < 3:
        return None
    xs = [math.log1p(max(t, 0.0)) for t, _, _ in pts]
    ws = recency_weights([t for t, _, _ in pts], at_t)
    ye = [math.log(max(v * n / 100.0, 0.5)) for _, v, n in pts]
    yn = [math.log(n) for _, _, n in pts]
    ae, be, _ = wls(xs, ye, ws)
    an, bn, _ = wls(xs, yn, ws)
    xt = math.log1p(max(x_target, 0.0))
    rate = 100.0 * math.exp((ae + be * xt) - (an + bn * xt))
    return logit(rate)


def fit_all_models(times, vals, ns, at_t, x_target, asymp, shrink_b=None,
                   cls=None):
    """Returns dict model -> median logit (falls back to trend when needed).
    When `cls` is given, the class prior day-7 slope is blended into the
    trend and logistic fits (see prior_slope)."""
    last = None
    for v in vals:
        if v is not None:
            last = v
    if last is None:
        last = 0.05
    xt_at = math.log1p(max(at_t, 0.0))
    xt_tg = math.log1p(max(x_target, 0.0))
    # Prior slope anchored to day 7 (168h), not to the case target: the
    # class's typical log-time slope is horizon-independent. Anchoring to
    # the case target makes the implied slope explode on short horizons
    # (practice cases: 90 min) and is only restrained by the cap.
    xt_d7 = math.log1p(168.0)
    if cls and PRIOR_STRENGTH > 0:
        ps_trend = prior_slope(cls, last, xt_at, xt_d7)
        ps_log = prior_slope(cls, last, xt_at, xt_d7, asymp)
        ps_log2 = prior_slope(cls, last, xt_at, xt_d7, 0.6 * asymp)
    else:
        ps_trend = ps_log = ps_log2 = None
    trend, _ = fit_trend(times, vals, at_t, x_target, shrink_b, ps_trend)
    out = {"trend": trend}
    lg1 = fit_logistic(times, vals, at_t, x_target, asymp, ps_log)
    lg2 = fit_logistic(times, vals, at_t, x_target, 0.6 * asymp, ps_log2)
    if lg1 is not None and lg2 is not None:
        out["logistic"] = 0.5 * (lg1 + lg2)
    elif lg1 is not None:
        out["logistic"] = lg1
    else:
        out["logistic"] = trend
    cnt = fit_counts(times, vals, ns, at_t, x_target)
    out["counts"] = cnt if cnt is not None else trend
    return out


def backtest_weights(reads, trophies, classes, at_t, x_target):
    """Refit each model on data truncated 12h/24h before the checkpoint and
    score it against the realised last read. Errors are averaged (the
    competition loss is a mean over trophies, so a median would hide the
    trophies that actually move). Returns (global weights, global mean
    errors per model, per-class weights, per-class mean errors)."""
    empty = (dict(PRIOR_W), {m: 0.6 for m in PRIOR_W}, {}, {})
    if len(reads) < 6:
        return empty
    last_rates = reads[-1]["rates"]
    errs = {"logistic": [], "trend": [], "counts": []}
    class_errs = {}
    used = 0
    for cut_h in (12.0, 24.0):
        cut = at_t - cut_h
        early = [r for r in reads if r["t"] <= cut]
        if len(early) < 3:
            continue
        # class-mean trend slopes on the truncated data (for shrinkage)
        slopes = {}
        for i, tr in enumerate(trophies):
            cls = classes[i][0]
            vals = [r["rates"][i] for r in early]
            ts = [r["t"] for r in early]
            _, b, _ = wls([math.log1p(max(t, 0.0)) for t in ts],
                          [logit(v) for v in vals],
                          recency_weights(ts, cut)) if len(ts) >= 2 else (0, 0, 0)
            slopes.setdefault(cls, []).append(b)
        class_mean = {c: sum(v) / len(v) for c, v in slopes.items() if len(v) >= 2}
        used += 1
        for i, tr in enumerate(trophies):
            if last_rates[i] is None:
                continue
            times = [r["t"] for r in early]
            vals = [r["rates"][i] for r in early]
            ns = [r["n"] for r in early]
            shrink = class_mean.get(classes[i][0])
            # predict the last read's rate from the truncated fit: the
            # evaluation point is the last read's time
            preds = fit_all_models(times, vals, ns, cut,
                                   reads[-1]["t"], classes[i][1], shrink,
                                   classes[i][0])
            y = logit(last_rates[i])
            for m, p in preds.items():
                e = abs(p - y)
                errs[m].append(e)
                class_errs.setdefault(classes[i][0], {}).setdefault(m, []).append(e)
    if not used:
        return empty

    def weights_from(mean_err):
        raw = {m: 1.0 / (mean_err[m] ** 2 + 0.04) for m in mean_err}
        tot = sum(raw.values())
        return {m: raw[m] / tot for m in raw}

    mean_err = {m: (sum(es) / len(es) if es else 1.0) for m, es in errs.items()}
    bt_w = weights_from(mean_err)
    w = {m: 0.5 * PRIOR_W[m] + 0.5 * bt_w[m] for m in PRIOR_W}
    class_w = {}
    class_mean_err = {}
    for cls, per_m in class_errs.items():
        ce = {m: (sum(es) / len(es) if es else 1.0) for m, es in per_m.items()}
        if len(per_m.get("trend", [])) < 6:
            continue
        class_mean_err[cls] = ce
        cb = weights_from(ce)
        class_w[cls] = {m: 0.4 * PRIOR_W[m] + 0.3 * bt_w[m] + 0.3 * cb[m]
                        for m in PRIOR_W}
    return w, mean_err, class_w, class_mean_err


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    levels = case.get("levels") or LEVELS
    if levels != LEVELS:
        raise ValueError("unexpected levels %r" % (levels,))
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_days = max(horizon_min / 1440.0, 0.0)
    at_dt = parse_time(case.get("at"))
    target_dt = parse_time(case.get("target"))
    at_t = hours_since_release(at_dt)
    target_t = hours_since_release(target_dt)
    if at_t is None:
        at_t = 0.0
    if target_t is None:
        target_t = at_t + horizon_min / 60.0

    classes = [classify(t) for t in trophies]
    reads = dedup_reads(history)
    flat = False
    if reads:
        flat = (len(reads) == 1 or
                all(r["rates"] == reads[0]["rates"] for r in reads))
    # Class-mean trend slopes (full history) for shrinkage.
    class_slopes = {}
    if reads:
        for i in range(len(trophies)):
            cls = classes[i][0]
            ts = [r["t"] for r in reads]
            vs = [r["rates"][i] for r in reads]
            if len(ts) >= 2:
                _, b, _ = wls([math.log1p(max(t, 0.0)) for t in ts],
                              [logit(v) for v in vs],
                              recency_weights(ts, at_t))
                class_slopes.setdefault(cls, []).append(b)
    class_mean = {c: sum(v) / len(v) for c, v in class_slopes.items()
                  if len(v) >= 2}

    weights, mean_err, class_w, class_mean_err = backtest_weights(
        reads, trophies, classes, at_t, target_t)
    base_spread = 0.45 * math.sqrt(horizon_days)
    bt_scale = math.sqrt(horizon_days / (18.0 / 24.0)) if horizon_days > 0 else 1.0
    bt_scale = min(bt_scale, 3.0)

    rows = []
    for i, tr in enumerate(trophies):
        tid = tr.get("id")
        cls, asymp = classes[i]
        times = [r["t"] for r in reads]
        vals = [r["rates"][i] for r in reads] if reads else []
        ns = [r["n"] for r in reads] if reads else []
        last = vals[-1] if vals else None
        if last is None:
            # Never observed: fall back to a small prior rate for the class.
            last = min(asymp * 0.02, 0.5)
        shrink = class_mean.get(cls)
        w_i = weights
        err_src = mean_err
        bt_err_i = sum(w_i[m] * err_src.get(m, mean_err[m]) for m in w_i)
        if reads:
            meds = fit_all_models(times, vals, ns, at_t, target_t, asymp,
                                  shrink, cls)
        else:
            lg = logit(last)
            meds = {"logistic": lg, "trend": lg, "counts": lg}
        med = sum(w_i[m] * meds[m] for m in w_i)
        dis = math.sqrt(sum(w_i[m] * (meds[m] - med) ** 2
                            for m in w_i))
        spread = math.sqrt(base_spread ** 2 + dis ** 2 +
                           (SPREAD_BT_FACTOR * bt_err_i * bt_scale) ** 2)
        spread = max(0.35, min(spread, SPREAD_CAP))
        if last <= 0.02 and PRIOR7.get(cls, 0.0) >= 50.0 * max(last, 0.01):
            # At the print floor with a class prior far above it, the
            # history carries no signal about the takeoff: widen the band
            # to cover the prior uncertainty instead of the (silent)
            # backtest error.
            spread = max(spread, FLOOR_SPREAD)
        if flat and at_t <= 48.0 and last < 99.0:
            # Stale page: a refresh is coming, tracked players will jump and
            # rates will fall. Shift the median down, widen the band.
            med += math.log(0.4)
            spread = max(spread, 1.1)
        if last >= 99.0:
            # Universal trophy: stays at the top with a narrow band.
            med = logit(99.99)
            spread = 0.15
        elif last >= 95.0:
            med = logit(last)
            spread = min(spread, 0.35)
        if med < logit(0.025) and last <= 0.05:
            # Do not claim a rate below what the page can print for a trophy
            # that has barely registered.
            med = logit(0.025)
        qs = [round4(clamp_pct(inv_logit(med + z * spread))) for z in Z]
        qs = sorted(qs)
        rows.append({"id": tid, "q": qs})
    rows.sort(key=lambda r: (r["id"] is None, r["id"]))
    return {"trophies": rows}


def main():
    in_dir = INPUT_DIR
    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    if not names:
        return 0
    for name in names:
        path = os.path.join(in_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        stem = name[:-3] if name.endswith(".in") else name
        out_path = os.path.join(out_dir, stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(forecast, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
