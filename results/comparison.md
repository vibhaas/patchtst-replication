# PatchTST replication: results vs. paper

Delta = (ours - paper) / paper on MSE. Negative means we did better. Notes list where a run deviates from the paper setup.

| Block | Run | MSE | MAE | Paper MSE | Paper MAE | Delta MSE | Notes |
|---|---|---|---|---|---|---|---|
| A | A/PatchTST42/ETTh1/96 | 0.376 | 0.401 | 0.375 | 0.399 | +0.4% | 28 ep, early_stop, 1.14 min |
| A | A/DLinear/ETTh1/96 | 0.438 | 0.452 | 0.375 | 0.399 | +16.8% | 8 ep, early_stop, 0.17 min |
| A | A/PatchTST42/ETTh1/192 | 0.414 | 0.422 | 0.414 | 0.421 | +0.0% | 29 ep, early_stop, 1.2 min |
| A | A/DLinear/ETTh1/192 | 0.456 | 0.457 | 0.405 | 0.416 | +12.6% | 6 ep, early_stop, 0.14 min |
| A | A/PatchTST42/ETTh1/336 | 0.432 | 0.437 | 0.431 | 0.436 | +0.3% | 14 ep, early_stop, 0.6 min |
| A | A/DLinear/ETTh1/336 | 0.476 | 0.474 | 0.439 | 0.443 | +8.5% | 10 ep, max_epochs, 0.22 min |
| A | A/PatchTST42/ETTh1/720 | 0.455 | 0.469 | 0.449 | 0.466 | +1.3% | 12 ep, early_stop, 0.52 min |
| A | A/DLinear/ETTh1/720 | 0.574 | 0.561 | 0.472 | 0.490 | +21.6% | 6 ep, early_stop, 0.16 min |
| A | A/PatchTST42/ETTh2/96 | 0.276 | 0.338 | 0.274 | 0.336 | +0.9% | 17 ep, early_stop, 0.73 min |
| A | A/DLinear/ETTh2/96 | 0.300 | 0.365 | 0.289 | 0.353 | +3.8% | 4 ep, early_stop, 0.12 min |
| A | A/PatchTST42/ETTh2/192 | 0.340 | 0.380 | 0.339 | 0.379 | +0.2% | 16 ep, early_stop, 0.72 min |
| A | A/DLinear/ETTh2/192 | 0.407 | 0.434 | 0.383 | 0.418 | +6.3% | 4 ep, early_stop, 0.09 min |
| A | A/PatchTST42/ETTh2/336 | 0.332 | 0.387 | 0.331 | 0.380 | +0.3% | 12 ep, early_stop, 0.5 min |
| A | A/DLinear/ETTh2/336 | 0.469 | 0.478 | 0.448 | 0.465 | +4.7% | 4 ep, early_stop, 0.07 min |
| A | A/PatchTST42/ETTh2/720 | 0.378 | 0.421 | 0.379 | 0.422 | -0.2% | 21 ep, early_stop, 0.65 min |
| A | A/DLinear/ETTh2/720 | 0.753 | 0.609 | 0.605 | 0.551 | +24.4% | 4 ep, early_stop, 0.07 min |
| A | A/PatchTST42/ETTm1/96 | 0.292 | 0.344 | 0.290 | 0.342 | +0.6% | training window stride 2; 17 ep, early_stop, 4.01 min |
| A | A/DLinear/ETTm1/96 | 0.300 | 0.345 | 0.299 | 0.343 | +0.5% | 10 ep, max_epochs, 1.34 min |
| A | A/PatchTST42/ETTm1/192 | 0.333 | 0.372 | 0.332 | 0.369 | +0.4% | training window stride 2; 15 ep, time_cap, 6.16 min |
| A | A/DLinear/ETTm1/192 | 0.338 | 0.370 | 0.335 | 0.365 | +0.8% | 10 ep, max_epochs, 1.79 min |
| A | A/PatchTST42/ETTm1/336 | 0.366 | 0.394 | 0.366 | 0.392 | -0.0% | training window stride 2; 16 ep, time_cap, 6.05 min |
| A | A/DLinear/ETTm1/336 | 0.370 | 0.388 | 0.369 | 0.386 | +0.2% | 10 ep, max_epochs, 1.44 min |
| A | A/PatchTST42/ETTm1/720 | 0.424 | 0.428 | 0.420 | 0.424 | +0.9% | training window stride 2; 20 ep, early_stop, 4.88 min |
| A | A/DLinear/ETTm1/720 | 0.426 | 0.421 | 0.425 | 0.421 | +0.3% | 9 ep, early_stop, 1.51 min |
| A | A/PatchTST42/ETTm2/96 | 0.163 | 0.252 | 0.165 | 0.255 | -1.0% | training window stride 2; 13 ep, early_stop, 3.07 min |
| A | A/DLinear/ETTm2/96 | 0.173 | 0.268 | 0.167 | 0.260 | +3.3% | 6 ep, early_stop, 0.29 min |
| A | A/PatchTST42/ETTm2/192 | 0.226 | 0.297 | 0.220 | 0.292 | +2.9% | training window stride 2; 10 ep, early_stop, 2.38 min |
| A | A/DLinear/ETTm2/192 | 0.226 | 0.299 | 0.224 | 0.303 | +0.9% | 10 ep, max_epochs, 0.49 min |
| A | A/PatchTST42/ETTm2/336 | 0.279 | 0.332 | 0.278 | 0.329 | +0.4% | training window stride 2; 10 ep, early_stop, 2.39 min |
| A | A/DLinear/ETTm2/336 | 0.305 | 0.363 | 0.281 | 0.342 | +8.6% | 4 ep, early_stop, 0.21 min |
| A | A/PatchTST42/ETTm2/720 | 0.366 | 0.385 | 0.367 | 0.385 | -0.2% | training window stride 2; 10 ep, early_stop, 2.43 min |
| A | A/DLinear/ETTm2/720 | 0.431 | 0.441 | 0.397 | 0.421 | +8.7% | 4 ep, early_stop, 0.23 min |
| B | B/CI-only/ETTh1/96 | 0.387 | 0.412 | 0.365 | 0.395 | +6.1% | 7 ep, early_stop, 4.72 min |
| B | B/P-only/ETTh1/96 | 0.416 | 0.437 | 0.416 | 0.438 | -0.1% | 30 ep, max_epochs, 0.36 min |
| B | B/Original/ETTh1/96 | 0.484 | 0.472 | 0.455 | 0.459 | +6.4% | 7 ep, early_stop, 1.26 min |
| B | B/CI-only/ETTh1/336 | 0.452 | 0.449 | 0.430 | 0.433 | +5.2% | 7 ep, early_stop, 4.62 min |
| B | B/P-only/ETTh1/336 | 0.499 | 0.491 | 0.484 | 0.480 | +3.1% | 8 ep, early_stop, 0.11 min |
| B | B/Original/ETTh1/336 | 0.526 | 0.504 | 0.514 | 0.503 | +2.3% | 6 ep, early_stop, 1.46 min |
| C | C/PatchTST/ETTh1/96/L96 | 0.395 | 0.410 | 0.393 | 0.408 | +0.6% | 27 ep, early_stop, 0.31 min |
| C | C/PatchTST/ETTh1/96/L192 | 0.384 | 0.403 | 0.382 | 0.401 | +0.6% | 22 ep, early_stop, 0.33 min |
| C | C/PatchTST/ETTh1/96/L512 | 0.375 | 0.405 |  |  |  | 15 ep, early_stop, 0.55 min |
| C | C/PatchTST/ETTh1/96/L720 | 0.375 | 0.407 | 0.376 | 0.408 | -0.2% | 16 ep, early_stop, 1.07 min |
| D | D/PatchTST42/weather/96 | 0.155 | 0.204 | 0.152 | 0.199 | +1.8% | training window stride 2; 21 ep, early_stop, 9.0 min |
| D | D/DLinear/weather/96 | 0.175 | 0.236 | 0.176 | 0.237 | -0.4% | 10 ep, max_epochs, 0.86 min |
| D | D/PatchTST42/weather/336 | 0.252 | 0.286 | 0.249 | 0.283 | +1.3% | training window stride 2; 15 ep, early_stop, 6.45 min |
| D | D/DLinear/weather/336 | 0.262 | 0.312 | 0.265 | 0.319 | -1.1% | 7 ep, early_stop, 0.68 min |
| E | E/PatchTST42-32ch/electricity/96 | 0.145 | 0.239 | 0.130 | 0.222 | +11.3% | trained on 32/321 channels; training window stride 2; 30 ep, max_epochs, 10.64 min |
| E | E/DLinear/electricity/96 | 0.142 | 0.240 | 0.140 | 0.237 | +1.3% | 10 ep, max_epochs, 1.87 min |
| X | X/PatchTST64/ETTh1/96 | 0.375 | 0.405 | 0.370 | 0.400 | +1.3% | 15 ep, early_stop, 0.55 min |
| X | X/PatchTST64/ETTh1/192 | 0.411 | 0.427 | 0.413 | 0.429 | -0.5% | 12 ep, early_stop, 0.45 min |
| X | X/PatchTST64/ETTh1/336 | 0.420 | 0.439 | 0.422 | 0.440 | -0.4% | 11 ep, early_stop, 0.4 min |
| X | X/PatchTST64/ETTh1/720 | 0.449 | 0.468 | 0.447 | 0.468 | +0.5% | 10 ep, early_stop, 0.36 min |
| X | X/PatchTST64/ETTh2/96 | 0.274 | 0.338 | 0.274 | 0.337 | +0.1% | 18 ep, early_stop, 0.66 min |
| X | X/PatchTST64/ETTh2/192 | 0.340 | 0.381 | 0.341 | 0.382 | -0.4% | 13 ep, early_stop, 0.49 min |
| X | X/PatchTST64/ETTh2/336 | 0.329 | 0.386 | 0.329 | 0.384 | +0.1% | 15 ep, early_stop, 0.55 min |
| X | X/PatchTST64/ETTh2/720 | 0.380 | 0.424 | 0.379 | 0.422 | +0.4% | 17 ep, early_stop, 0.61 min |
| X | X/PatchTST42/weather/192 | 0.201 | 0.246 | 0.197 | 0.243 | +2.0% | training window stride 2; 20 ep, early_stop, 8.57 min |
| X | X/PatchTST42/weather/720 | 0.320 | 0.335 | 0.320 | 0.335 | +0.1% | training window stride 2; 19 ep, early_stop, 8.21 min |
| X | X/PatchTST42-full/electricity/96 | 0.137 | 0.234 | 0.130 | 0.222 | +5.0% | 11 ep, time_cap, 60.45 min |
| F | F/ss-linprobe/ETTh1/96 | 0.368 | 0.397 | 0.371 | 0.400 | -0.9% | linear_probe |
| F | F/ss-finetune/ETTh1/96 | 0.370 | 0.400 | 0.366 | 0.397 | +1.2% | finetune |
| F | F/ss-linprobe/ETTh1/336 | 0.445 | 0.457 | 0.445 | 0.446 | +0.1% | linear_probe |
| F | F/ss-finetune/ETTh1/336 | 0.428 | 0.439 | 0.450 | 0.456 | -4.9% | finetune |

## Ablation on ETTh1 (paper Table 10): MSE ours / paper

| Horizon | P+CI (full) | CI only | P only | Original |
|---|---|---|---|---|
| 96 | 0.376 / 0.375 | 0.387 / 0.365 | 0.416 / 0.416 | 0.484 / 0.455 |
| 336 | 0.432 / 0.431 | 0.452 / 0.430 | 0.499 / 0.484 | 0.526 / 0.514 |

## Look-back window on ETTh1, horizon 96 (paper Table 9): MSE ours / paper

| L | ours | paper |
|---|---|---|
| 96 | 0.395 | 0.393 |
| 192 | 0.384 | 0.382 |
| 336 | 0.376 | 0.375 |
| 512 | 0.375 | n/a |
| 720 | 0.375 | 0.376 |
