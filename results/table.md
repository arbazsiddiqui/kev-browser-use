# kev-browser-use eval results

## dataset: 1k

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| random | 0.105 | 0.355 | 0.036 | 0.936 | 0.01 | 0.01 | 1000 |
| first | 0.102 | 0.822 | 0.088 | 1.796 | 0.00 | 0.00 | 1000 |
| lexical | 0.174 | 0.740 | 0.155 | 0.983 | 0.05 | 0.05 | 1000 |
| laya | 0.107 | 0.334 | 0.038 | 0.979 | 41.92 | 40.02 | 1000 |
| jev | 0.479 | 0.827 | 0.435 | 0.731 | 1107.00 | 1006.00 | 1000 |
| nanojev | 0.131 | 0.164 | 0.042 | 0.918 | 258.58 | 244.85 | 1000 |
| decider | 0.330 | 0.625 | 0.247 | 0.867 | 170.04 | 138.98 | 1000 |
| kev | 0.387 | 0.787 | 0.325 | 0.756 | 439.31 | 427.97 | 1000 |
| nimble | 0.385 | 0.596 | 0.262 | 0.869 | 990.67 | 979.25 | 1000 |

## dataset: 1k-exp2-modernbert-m2w

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: 1k-exp3-mindact-m2w

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: 1k-final-kev06-mix

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| kev | 0.361 | 0.836 | 0.322 | 0.838 | 79.66 | 79.51 | 1000 |

## dataset: 1k-final-mindact-mix

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: 1k-kev06

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| kev | 0.206 | 0.674 | 0.141 | 1.057 | 78.45 | 78.17 | 1000 |

## dataset: 1k-kev06-ft-exp4

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| kev | 0.348 | 0.797 | 0.299 | 0.788 | 78.59 | 78.20 | 1000 |

## dataset: 1k-kev08

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| kev | 0.205 | 0.645 | 0.128 | 0.923 | 100.46 | 99.63 | 1000 |

## dataset: 1k-laya-m2w

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| laya | 0.117 | 0.632 | 0.077 | 1.070 | 47.13 | 47.09 | 1000 |

## dataset: 1k-mindact-compact

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: 1k-mindact-mindact

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: 1k-mindact-native

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: 1k-openjev-deberta

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: 1k-semif-qwen35-4b

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| semif | 0.384 | 0.752 | 0.300 | 0.849 | 298.01 | 289.44 | 1000 |

## dataset: 1k-verdict

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: 1k_compact

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| laya | 0.115 | 0.312 | 0.037 | 0.960 | 42.16 | 40.05 | 1000 |

## dataset: 1k_native

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| random | 0.060 | 0.340 | 0.020 | 0.928 | 0.01 | 0.01 | 50 |
| first | 0.100 | 0.780 | 0.040 | 1.800 | 0.00 | 0.00 | 50 |
| lexical | 0.140 | 0.660 | 0.120 | 0.985 | 0.05 | 0.05 | 50 |
| laya | 0.120 | 0.340 | 0.060 | 1.011 | 103.15 | 40.24 | 50 |
| nanojev | 0.160 | 0.100 | 0.040 | 0.920 | 271.28 | 254.40 | 50 |
| decider | 0.300 | 0.660 | 0.180 | 0.922 | 503.25 | 211.95 | 50 |
| kev | 0.280 | 0.800 | 0.260 | 0.843 | 1473.26 | 426.42 | 50 |
| nimble | 0.280 | 0.520 | 0.200 | 1.071 | 1693.25 | 968.93 | 50 |

## dataset: smoke-agy-flash38high

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke-agy-gptoss120b

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke-agy-opus46

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke-agy-pro31high

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke-agy-sonnet46

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke-kev06

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| kev | 0.220 | 0.700 | 0.140 | 0.979 | 78.98 | 78.29 | 50 |

## dataset: smoke-kev06-ft-exp4

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| kev | 0.280 | 0.780 | 0.200 | 0.841 | 80.92 | 79.19 | 50 |

## dataset: smoke-kev08

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| kev | 0.140 | 0.660 | 0.060 | 0.969 | 318.26 | 99.53 | 50 |

## dataset: smoke-laya-m2w

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| laya | 0.080 | 0.540 | 0.060 | 1.095 | 59.27 | 46.49 | 50 |

## dataset: smoke-mindact-compact

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke-mindact-mindact

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke-mindact-native

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke-openjev-deberta

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke-or-nemotron3super

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke-or-nexn25pro

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke-semif-qwen35-4b

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
| semif | 0.280 | 0.740 | 0.140 | 0.939 | 1159.13 | 325.95 | 50 |

## dataset: smoke-verdict

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|

## dataset: smoke_native

| adapter | element acc | op acc | step success | brier | latency mean (ms) | latency p50 (ms) | n |
|---|---|---|---|---|---|---|---|
