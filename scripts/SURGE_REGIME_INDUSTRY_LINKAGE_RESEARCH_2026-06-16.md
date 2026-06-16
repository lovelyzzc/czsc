# Surge Regime industry-linkage research (2026-06-16)

First-stage test of industry/sector linkage for the delay5 surge-regime strategy.
This uses stock_basic.industry plus the existing daily panel; it does not change live behavior.

## Pass Criterion

- OOS 10-slot trades >= 60
- OOS trade excess mean >= baseline + 1.5 pp
- OOS trade excess median >= baseline median
- OOS t >= 2
- IS trade excess mean >= baseline IS mean

## Universe

```json
{
  "universe_rows": 98540,
  "base_candidates": 3185,
  "base_oos_candidates": 1908,
  "industries_total": 111,
  "industries_base": 109,
  "first_test_year": 2024,
  "slots": 10
}
```

## Gate Results
| gate | passed | cand | oos_n | oos_mean | oos_med | oos_t | is_mean | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | False | 3185 | 210 | 7.58 | -0.19 | 2.96 | 0.91 | baseline |
| ind_ret20_gt0 | False | 3019 | 200 | 7.76 | -0.45 | 2.88 | 1.25 | mean_gain,median |
| ind_ret20_rank_top50 | False | 2525 | 209 | 5.19 | -0.89 | 3.61 | 0.91 | mean_gain,median |
| ind_ret20_rank_top70 | False | 1948 | 204 | 4.73 | -0.73 | 3.41 | 0.23 | mean_gain,median,is_mean |
| ind_high20_gt012 | False | 2110 | 197 | 4.23 | -1.51 | 3.31 | -0.22 | mean_gain,median,is_mean |
| ind_high20_gt_market | False | 1402 | 200 | 6.12 | -1.62 | 2.94 | 0.06 | mean_gain,median,is_mean |
| ind_limit_up_rank_top70 | False | 1703 | 200 | 4.2 | -0.72 | 3.85 | -0.44 | mean_gain,median,is_mean |
| ind_struct_density_top70 | False | 2372 | 202 | 4.82 | -0.35 | 3.67 | 1.5 | mean_gain,median |
| ind_composite_top50 | False | 2770 | 210 | 4.85 | -0.57 | 3.73 | 0.85 | mean_gain,median,is_mean |
| ind_composite_top70 | False | 1421 | 200 | 3.68 | -1.24 | 3.25 | -0.5 | mean_gain,median,is_mean |

## Conclusion

No industry-linkage gate passed the pre-declared first-stage screen. Industry information shows descriptive structure but is not yet a deployable filter for delay5.

## OOS Candidate Buckets

### ind_ret20_mean_rank
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (0.00801, 0.495] | 395 | 4.64 | -0.93 | 2.95 | 16.1 |
| (0.495, 0.694] | 378 | 3.17 | -0.12 | 4.14 | 17.3 |
| (0.694, 0.811] | 387 | 2.35 | -1.03 | 3.14 | 17.8 |
| (0.811, 0.91] | 372 | 3.35 | -0.65 | 4.23 | 17.2 |
| (0.91, 1.0] | 376 | 3.9 | -0.41 | 4.45 | 21.2 |

### ind_high20_ratio_rank
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (0.016999999999999998, 0.288] | 389 | 3.34 | -0.86 | 2.45 | 15.2 |
| (0.288, 0.454] | 374 | 3.39 | -0.81 | 3.96 | 17.8 |
| (0.454, 0.631] | 389 | 3.9 | -0.6 | 3.69 | 17.0 |
| (0.631, 0.809] | 374 | 3.74 | -0.22 | 4.6 | 17.3 |
| (0.809, 1.0] | 382 | 3.06 | -0.67 | 3.59 | 22.4 |

### ind_limit_up_ratio_rank
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (0.161, 0.36] | 447 | 4.77 | -0.63 | 4.48 | 19.3 |
| (0.36, 0.64] | 326 | 3.48 | -0.82 | 2.16 | 14.4 |
| (0.64, 0.748] | 396 | 2.21 | -1.03 | 3.09 | 14.3 |
| (0.748, 0.82] | 362 | 3.5 | -0.52 | 4.29 | 20.5 |
| (0.82, 1.0] | 377 | 3.3 | -0.38 | 4.37 | 20.6 |

### ind_struct_density_rank
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (0.368, 0.658] | 390 | 5.1 | -0.74 | 3.11 | 15.8 |
| (0.658, 0.755] | 374 | 2.01 | -0.6 | 3.33 | 16.0 |
| (0.755, 0.829] | 413 | 3.13 | -0.89 | 4.15 | 18.9 |
| (0.829, 0.901] | 368 | 3.58 | -0.48 | 4.62 | 21.3 |
| (0.901, 1.0] | 363 | 3.59 | -0.54 | 3.92 | 17.5 |

### ind_composite_rank
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (0.264, 0.538] | 394 | 4.56 | -1.25 | 2.86 | 18.2 |
| (0.538, 0.631] | 375 | 3.2 | -0.28 | 4.5 | 15.5 |
| (0.631, 0.704] | 380 | 2.95 | -0.66 | 3.42 | 15.3 |
| (0.704, 0.786] | 377 | 3.23 | -0.14 | 4.31 | 17.2 |
| (0.786, 0.995] | 382 | 3.45 | -0.67 | 4.24 | 23.3 |
