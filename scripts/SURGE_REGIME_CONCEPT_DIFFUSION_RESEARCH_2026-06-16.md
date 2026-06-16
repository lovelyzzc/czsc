# Surge Regime concept-diffusion research (2026-06-16)

First-stage test of fine-grained 同花顺概念 diffusion for the delay5 surge-regime strategy.
It uses ths_index(type=N) membership plus the existing daily panel; it does not change live behavior.

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
  "all_concepts": 412,
  "fine_concepts": 273,
  "fine_member_links": 19733,
  "base_candidates_with_concepts": 3185,
  "first_test_year": 2024,
  "slots": 10
}
```

## Gate Results
| gate | passed | cand | oos_n | oos_mean | oos_med | oos_t | is_mean | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | False | 3185 | 210 | 7.58 | -0.19 | 2.96 | 0.91 | baseline |
| cpt_composite_max_top70 | False | 1767 | 195 | 5.53 | -0.76 | 3.64 | 0.24 | mean_gain,median,is_mean |
| cpt_composite_max_top85 | False | 592 | 164 | 3.82 | -1.0 | 2.68 | 0.0 | mean_gain,median,is_mean |
| cpt_composite_mean_top60 | False | 1454 | 194 | 6.02 | -0.73 | 3.88 | -0.24 | mean_gain,median,is_mean |
| cpt_hot70_count_ge2 | False | 967 | 175 | 8.0 | 0.39 | 3.51 | 0.05 | mean_gain,is_mean |
| cpt_hot70_share_ge50 | False | 881 | 178 | 5.48 | -0.93 | 2.57 | -0.67 | mean_gain,median,is_mean |
| cpt_density_max_top80 | False | 2311 | 210 | 5.5 | -0.08 | 4.4 | 0.47 | mean_gain,is_mean |
| cpt_limit_up_max_top80 | False | 1580 | 195 | 3.56 | -1.16 | 2.88 | 0.11 | mean_gain,median,is_mean |
| cpt_ret20_max_top80 | False | 1792 | 198 | 6.23 | -0.88 | 4.1 | 0.49 | mean_gain,median,is_mean |
| cpt_high20_max_top80 | False | 1344 | 192 | 5.88 | -0.72 | 3.79 | 0.05 | mean_gain,median,is_mean |

## Conclusion

No concept-diffusion gate passed the pre-declared first-stage screen. Fine-grained concepts show descriptive heat but are not yet a deployable delay5 filter.

## OOS Candidate Buckets

### cpt_composite_rank_max
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (0.197, 0.587] | 366 | 1.87 | -1.18 | 2.79 | 17.1 |
| (0.587, 0.685] | 359 | 3.89 | -0.63 | 4.29 | 15.8 |
| (0.685, 0.763] | 362 | 2.06 | -1.08 | 2.81 | 15.7 |
| (0.763, 0.842] | 362 | 4.44 | -0.28 | 4.08 | 19.5 |
| (0.842, 0.994] | 363 | 3.87 | -0.45 | 4.14 | 20.8 |

### cpt_composite_rank_mean
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (0.197, 0.488] | 363 | 2.93 | -0.72 | 3.55 | 20.1 |
| (0.488, 0.561] | 362 | 3.4 | -0.91 | 4.06 | 14.4 |
| (0.561, 0.621] | 362 | 2.28 | -0.1 | 3.6 | 12.9 |
| (0.621, 0.7] | 362 | 4.39 | -0.78 | 3.59 | 19.7 |
| (0.7, 0.948] | 363 | 3.11 | -0.69 | 4.05 | 21.8 |

### hot70_count
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (-0.001, 1.0] | 1407 | 3.26 | -0.75 | 6.23 | 16.9 |
| (1.0, 2.0] | 247 | 5.36 | 0.06 | 3.47 | 19.5 |
| (2.0, 13.0] | 254 | 2.91 | -0.73 | 3.0 | 21.9 |

### hot70_share
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (-0.001, 0.25] | 1088 | 3.12 | -0.73 | 6.62 | 16.3 |
| (0.25, 0.5] | 402 | 2.68 | -0.49 | 4.19 | 17.9 |
| (0.5, 1.0] | 322 | 4.24 | -0.9 | 3.23 | 22.6 |

### cpt_struct_density_rank_max
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (0.217, 0.775] | 367 | 3.81 | -1.17 | 3.3 | 18.5 |
| (0.775, 0.882] | 362 | 2.96 | -0.76 | 3.46 | 15.6 |
| (0.882, 0.941] | 368 | 2.75 | -0.67 | 3.94 | 20.9 |
| (0.941, 0.974] | 353 | 3.21 | -0.35 | 3.9 | 18.3 |
| (0.974, 1.0] | 362 | 3.39 | -0.55 | 4.26 | 15.6 |

### cpt_limit_up_ratio_rank_max
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (0.0691, 0.601] | 366 | 2.44 | -1.21 | 3.27 | 16.3 |
| (0.601, 0.747] | 361 | 4.31 | -0.32 | 5.05 | 20.6 |
| (0.747, 0.841] | 363 | 3.13 | -0.45 | 2.9 | 15.8 |
| (0.841, 0.93] | 373 | 2.61 | -0.73 | 3.52 | 16.8 |
| (0.93, 1.0] | 349 | 3.67 | -0.93 | 3.88 | 19.7 |

### cpt_ret20_mean_rank_max
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (0.0138, 0.609] | 364 | 2.33 | -0.73 | 3.36 | 19.1 |
| (0.609, 0.793] | 368 | 3.13 | -0.62 | 4.04 | 16.1 |
| (0.793, 0.892] | 355 | 4.12 | -0.46 | 3.59 | 16.2 |
| (0.892, 0.963] | 385 | 2.2 | -0.95 | 3.14 | 16.1 |
| (0.963, 1.0] | 340 | 4.49 | -0.68 | 4.35 | 21.7 |

### cpt_high20_ratio_rank_max
| bucket | n | mean | median | t | t1_rate |
| --- | --- | --- | --- | --- | --- |
| (0.0138, 0.504] | 363 | 1.54 | -1.03 | 2.45 | 16.2 |
| (0.504, 0.694] | 362 | 3.08 | -0.86 | 4.13 | 16.9 |
| (0.694, 0.823] | 365 | 4.06 | -0.45 | 4.52 | 19.9 |
| (0.823, 0.924] | 359 | 4.36 | -0.37 | 3.66 | 16.0 |
| (0.924, 1.0] | 363 | 3.08 | -0.89 | 3.69 | 19.9 |
