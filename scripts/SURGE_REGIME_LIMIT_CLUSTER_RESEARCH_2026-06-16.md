# Surge Regime limit-up cluster detector research (2026-06-16)

First-stage label/coverage test for a complementary detector targeting limit-up-dense main-up moves.
No existing strategy threshold is changed; this is not a trading backtest.

## Pre-declared pass criterion

- OOS deduped events >= 200
- OOS t1_px30 lift >= 1.5x versus the primary universe
- OOS t1-positive events not covered by current delay5 baseline >= 50

## Universe

Primary universe: regime in {4,5,6}, not already in an FSM surge event, non-null t1_px30.

```json
{
  "primary_rows": 1688777,
  "primary_symbols": 4594,
  "delay5_base_rows": 3185,
  "cooldown_bars": 20,
  "first_test_year": 2024
}
```

Base t1_px30 rates:

```json
{
  "ALL": 12.4,
  "IS": 8.2,
  "OOS": 16.2
}
```

## Detector Verdicts
| detector | passed | oos_n | oos_rate% | lift | incr_t1 | delay5_cover% | fwd40_mean% | 000636_hits | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lc_breakout_v1 | False | 2743 | 20.1 | 1.24 | 551 | 0.2 | 18.48 | 0 | lift |
| lc_pullback_v1 | False | 3044 | 20.3 | 1.25 | 618 | 0.2 | 18.9 | 0 | lift |
| lc_dense10_v1 | False | 2851 | 20.2 | 1.25 | 576 | 0.1 | 18.62 | 0 | lift |
| lc_near_board_v1 | False | 1180 | 19.2 | 1.18 | 226 | 0.3 | 17.94 | 0 | lift |

## Conclusion

No detector passed the pre-declared first-stage gate. Do not move to portfolio mirror-backtest without redesign.

## Post-Hoc Diagnostic: Longer Post-Board Memory

After v1 missed 000636.SZ, three long-memory post-board base rules were tested as diagnostics only. They are not eligible for the pre-declared pass verdict.
| diagnostic | oos_n | oos_rate% | lift | fwd40_mean% | 000636_hits |
| --- | --- | --- | --- | --- | --- |
| diag_post_board_base_loose | 11769 | 16.2 | 1.0 | 16.54 | 8 |
| diag_post_board_base_tight | 8255 | 15.2 | 0.94 | 15.75 | 7 |
| diag_post_board_reclaim | 12378 | 16.4 | 1.01 | 16.69 | 9 |

## Detector Details

### lc_breakout_v1

| segment | n | symbols | t1_px30_rate_pct | lift | fwd40_mean_pct | fwd40_median_pct | fwd20_mean_pct | delay5_cover_pct | incremental_t1_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 4894 | 2292 | 17.0 | 1.37 | 16.42 | 9.56 | -2.74 | 0.1 | 829 |
| IS | 2151 | 1389 | 12.9 | 1.58 | 13.8 | 8.24 | -3.83 | 0.0 | 278 |
| OOS | 2743 | 1702 | 20.1 | 1.24 | 18.48 | 10.12 | -1.88 | 0.2 | 551 |

Yearly:
| year | n | t1_px30_rate_pct | delay5_cover_pct | fwd40_mean_pct |
| --- | --- | --- | --- | --- |
| 2021 | 264 | 15.2 | 0.0 | 15.64 |
| 2022 | 1238 | 13.2 | 0.0 | 14.45 |
| 2023 | 649 | 11.6 | 0.0 | 11.81 |
| 2024 | 1208 | 20.9 | 0.1 | 19.13 |
| 2025 | 1104 | 18.4 | 0.4 | 17.44 |
| 2026 | 431 | 22.5 | 0.2 | 19.32 |

### lc_pullback_v1

| segment | n | symbols | t1_px30_rate_pct | lift | fwd40_mean_pct | fwd40_median_pct | fwd20_mean_pct | delay5_cover_pct | incremental_t1_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 5440 | 2312 | 17.4 | 1.4 | 17.04 | 9.99 | -1.45 | 0.1 | 946 |
| IS | 2396 | 1449 | 13.7 | 1.68 | 14.67 | 8.75 | -2.7 | 0.0 | 328 |
| OOS | 3044 | 1745 | 20.3 | 1.25 | 18.9 | 10.82 | -0.47 | 0.2 | 618 |

Yearly:
| year | n | t1_px30_rate_pct | delay5_cover_pct | fwd40_mean_pct |
| --- | --- | --- | --- | --- |
| 2021 | 235 | 13.6 | 0.0 | 15.58 |
| 2022 | 1455 | 14.6 | 0.1 | 15.47 |
| 2023 | 706 | 12.0 | 0.0 | 12.74 |
| 2024 | 1338 | 21.5 | 0.1 | 19.79 |
| 2025 | 1286 | 18.9 | 0.3 | 17.66 |
| 2026 | 420 | 20.7 | 0.2 | 19.84 |

### lc_dense10_v1

| segment | n | symbols | t1_px30_rate_pct | lift | fwd40_mean_pct | fwd40_median_pct | fwd20_mean_pct | delay5_cover_pct | incremental_t1_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 5054 | 2288 | 17.4 | 1.4 | 16.89 | 9.72 | -2.36 | 0.1 | 878 |
| IS | 2203 | 1412 | 13.7 | 1.67 | 14.65 | 8.44 | -3.58 | 0.0 | 302 |
| OOS | 2851 | 1708 | 20.2 | 1.25 | 18.62 | 10.05 | -1.42 | 0.1 | 576 |

Yearly:
| year | n | t1_px30_rate_pct | delay5_cover_pct | fwd40_mean_pct |
| --- | --- | --- | --- | --- |
| 2021 | 226 | 14.2 | 0.0 | 16.1 |
| 2022 | 1306 | 14.3 | 0.0 | 15.81 |
| 2023 | 671 | 12.4 | 0.0 | 11.91 |
| 2024 | 1244 | 21.7 | 0.1 | 20.25 |
| 2025 | 1208 | 17.9 | 0.2 | 16.79 |
| 2026 | 399 | 22.6 | 0.0 | 19.06 |

### lc_near_board_v1

| segment | n | symbols | t1_px30_rate_pct | lift | fwd40_mean_pct | fwd40_median_pct | fwd20_mean_pct | delay5_cover_pct | incremental_t1_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 2128 | 1347 | 17.1 | 1.38 | 16.17 | 9.44 | -3.18 | 0.1 | 363 |
| IS | 948 | 717 | 14.5 | 1.76 | 13.96 | 7.85 | -4.67 | 0.0 | 137 |
| OOS | 1180 | 898 | 19.2 | 1.18 | 17.94 | 10.03 | -1.99 | 0.3 | 226 |

Yearly:
| year | n | t1_px30_rate_pct | delay5_cover_pct | fwd40_mean_pct |
| --- | --- | --- | --- | --- |
| 2021 | 116 | 19.8 | 0.0 | 16.76 |
| 2022 | 541 | 12.8 | 0.0 | 13.21 |
| 2023 | 291 | 15.5 | 0.0 | 14.24 |
| 2024 | 397 | 17.9 | 0.3 | 16.39 |
| 2025 | 587 | 18.2 | 0.3 | 17.48 |
| 2026 | 196 | 24.5 | 0.0 | 22.41 |
