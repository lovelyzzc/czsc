# Surge Regime sell-side warning research (2026-06-16)

This report tests whether FSM sell states 9/10 have value as a holding risk warning.
Positive avoid-excess means selling at next open avoided underperformance versus same-date/same-amount-decile median controls.

## Protocol

- Sell warning: first transition into Divergence(9) or Breakdown(10) after a non-sell state.
- Primary population: previous state in {5,6,7,8}.
- Returns: next-open to next-open over 5/10/20 bars.
- Pass criterion: primary OOS horizon-10 avoid mean > 0, median > 0, t >= 2, and IS mean > 0.

## Verdict

```json
{
  "passed": false,
  "failed": [
    "oos_mean",
    "oos_t",
    "is_mean"
  ]
}
```

Event counts:

```json
{
  "events": 194672,
  "primary_events": 187860,
  "oos_primary_events": 102253
}
```

## Primary Results
| segment | horizon | n | event_mean_pct | event_median_pct | control_median_mean_pct | avoid_mean_pct | avoid_median_pct | avoid_t | avoid_positive_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 5 | 186719 | 0.13 | -0.33 | -0.64 | -0.77 | -0.0 | -52.82 | 49.9 |
| ALL | 10 | 185930 | 0.31 | -0.45 | -0.93 | -1.24 | -0.01 | -59.37 | 49.8 |
| ALL | 20 | 183949 | 0.58 | -0.85 | -1.43 | -2.0 | -0.08 | -68.93 | 49.5 |
| IS | 5 | 85607 | 0.06 | -0.38 | -0.65 | -0.71 | -0.02 | -34.57 | 49.6 |
| IS | 10 | 85607 | -0.03 | -0.69 | -1.08 | -1.05 | -0.06 | -37.07 | 49.4 |
| IS | 20 | 85607 | -0.53 | -1.6 | -2.16 | -1.63 | -0.13 | -42.74 | 49.1 |
| OOS | 5 | 101112 | 0.2 | -0.28 | -0.63 | -0.83 | 0.01 | -39.94 | 50.2 |
| OOS | 10 | 100323 | 0.59 | -0.23 | -0.8 | -1.4 | 0.01 | -46.38 | 50.1 |
| OOS | 20 | 98342 | 1.54 | -0.16 | -0.79 | -2.33 | -0.03 | -54.12 | 49.7 |

## Breakdown vs Divergence

| group | segment | n | event_mean_pct | event_median_pct | avoid_mean_pct | avoid_median_pct | avoid_t | avoid_positive_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_breakdown | IS | 80914 | 0.02 | -0.65 | -1.07 | -0.07 | -36.93 | 49.3 |
| primary_breakdown | OOS | 94264 | 0.63 | -0.19 | -1.37 | 0.01 | -44.52 | 50.2 |
| primary_divergence | IS | 4693 | -0.91 | -1.54 | -0.75 | 0.09 | -5.51 | 50.7 |
| primary_divergence | OOS | 6059 | 0.08 | -1.02 | -1.77 | -0.09 | -13.02 | 49.4 |

## Yearly Primary Horizon-10

| year | n_h10 | h5 | h10 | h20 | h10_t |
| --- | --- | --- | --- | --- | --- |
| 2021 | 5986 | -0.88 | -1.14 | -1.59 | -9.48 |
| 2022 | 37214 | -0.83 | -1.14 | -1.64 | -24.87 |
| 2023 | 42407 | -0.58 | -0.95 | -1.62 | -25.97 |
| 2024 | 36204 | -0.61 | -0.98 | -1.5 | -19.42 |
| 2025 | 46224 | -0.9 | -1.52 | -2.45 | -36.11 |
| 2026 | 17895 | -1.06 | -1.92 | -3.85 | -24.35 |
