# Implementation log

Replication of *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*
(PatchTST, Nie et al., ICLR 2023).

## 2026-10-25: Scoping

- Read the paper. Main ideas: patching (subseries of 16 steps, stride 8, as Transformer tokens) and
  channel-independence (each variable forecast separately by one shared Transformer), plus masked
  self-supervised pre-training on patches.
- Goal: reproduce the main claims, not the full result tables. A full replication is 64 supervised
  runs plus pre-training, and the large datasets (Traffic, Electricity) need a 48 GB GPU in the
  paper's setup.
- Constraints: one 6 GB RTX 5050 and about 3 hours of training in total.
- The datasets are small (about 265 MB for all eight), so data size is not the bottleneck; compute is.

## 2026-10-25: Setup

- Used the authors' official code (github.com/yuqinie98/PatchTST, Apache-2.0) rather than a
  re-implementation, so differences can be attributed to the training budget and not to the model code.
- The code dates from 2023 (PyTorch 1.11). Made small compatibility fixes for current NumPy, pandas and
  PyTorch, and removed hard-coded data paths and CUDA assumptions.
- Added options to fit the budget: a per-run time limit, training on a subset of channels (test still
  on all channels), using every k-th training window, and skipping the per-epoch test pass.
- Implemented a channel-mixing version of PatchTST for the ablation study (Appendix A.4.3), since the
  official repo does not include one.
- Extracted the paper's result tables (3, 9, 10, 12) into a CSV for automatic comparison.
- All changes to the official code are listed in `UPSTREAM_PATCHES.md`.

## 2026-10-25: Experiment plan

Priority order, so the most important results finish first if time runs out:

1. PatchTST/42 vs. DLinear on the four ETT datasets, all horizons (Table 3).
2. Ablation of patching and channel-independence on ETTh1 (Table 10).
3. Look-back window 96 to 720 on ETTh1 (Table 9).
4. Weather, horizons 96 and 336.
5. Electricity trained on 32 of 321 channels and tested on all, to see how much a channel-independent
   model needs all channels during training.
6. Self-supervised pre-training on ETTh1, then linear probing and fine-tuning (Table 12).

Deviations from the paper, chosen for the budget: fewer epochs (30 max with early stopping instead of
100), every 2nd training window on the larger datasets, mixed precision on Weather and Electricity,
and shorter pre-training (20 epochs instead of 100).

## 2026-10-25: Main replication on the RTX 5050 laptop

- I installed PyTorch 2.11.0 with CUDA 12.8 in a local virtual environment. The RTX 5050 Laptop GPU
  reported 7.5 GiB of memory, and the CUDA kernel and autocast check passed. I downloaded the ETT,
  Weather and Electricity CSVs from the configured mirror.
- The smoke plan exposed two setup problems that I fixed before the main runs. The source copy omitted
  the four official self-supervised data loader files; I restored them from the pinned upstream commit
  and applied the documented pandas 3 compatibility change. The unpatched, channel-independent model
  exceeded GPU memory at batch size 128, so I set its smoke and block B batch size to 4. I also used
  batch size 4 for the unpatched channel-mixing variant in block B. All seven smoke paths then passed.
- I ran all 53 main jobs in 100.8 minutes with no failures. The supervised jobs used `cuda:0`.
  Twenty-five of the 32 block A jobs early-stopped, five reached their epoch limit, and two reached
  their six-minute cap. The ETTm1/192 and ETTm1/336 PatchTST runs used their best checkpoints at the cap.
- PatchTST/42 matched the paper closely on all 16 ETT settings: every MSE was within 3% of the
  reported value. At horizon 96, MSE was 0.376 on ETTh1, 0.276 on ETTh2, 0.292 on ETTm1 and 0.163
  on ETTm2. The corresponding DLinear results were 0.438, 0.300, 0.300 and 0.173, so PatchTST won
  all four. DLinear diverged more from the paper on several ETTh settings under the shorter schedule.
- The ablation supported both design choices. On ETTh1 at horizon 96, MSE was 0.376 with patching and
  channel independence, 0.387 with channel independence alone, 0.416 with patching alone and 0.484
  with neither. The horizon-336 ordering was the same (0.432, 0.452, 0.499, 0.526).
- For ETTh1 horizon 96, look-back lengths 96, 192, 336, 512 and 720 gave MSE 0.395, 0.384, 0.376,
  0.375 and 0.375. Most of the gain was present by length 336; the longest windows were nearly flat.
- Weather/96 PatchTST reached 0.155 MSE / 0.204 MAE versus the paper's 0.152 / 0.199. Weather/336
  reached 0.252 / 0.286 versus 0.249 / 0.283. DLinear was worse at both horizons.
- The Electricity model trained on 32 of 321 channels and was tested on all 321. It reached 0.145
  MSE / 0.239 MAE, compared with the paper's full-channel PatchTST result of 0.130 / 0.222. It did
  not beat the full-channel DLinear run (0.142 / 0.240), but confirmed that the shared model can
  forecast channels it did not see during training.
- I ran 20 epochs of ETTh1 masked pre-training, then linear probing and fine-tuning. At horizon 96,
  MSE was 0.368 for the probe and 0.370 for fine-tuning; the paper reports 0.371 and 0.366. At
  horizon 336, MSE was 0.445 and 0.428; the paper reports 0.445 and 0.450. These runs used the
  shortened schedules documented above.

## 2026-10-25: Extra runs

- The main plan finished faster than the estimated three-hour budget, so I ran the optional plan
  with a 65-minute start budget. Eleven of twelve extra jobs completed in 82.9 minutes with no
  failures. The Traffic job was not started because the budget had expired. Main and extra training
  together took 183.7 minutes.
- PatchTST/64 on ETTh1 and ETTh2 covered all four horizons for each dataset. All eight MSE values
  were within 1.3% of the paper. At horizon 96, ETTh1 reached 0.375 and ETTh2 reached 0.274.
- The added Weather horizons reached 0.201 MSE / 0.246 MAE at 192 and 0.320 / 0.335 at 720,
  compared with the paper's 0.197 / 0.243 and 0.320 / 0.335.
- I trained PatchTST on all 321 Electricity channels and all training windows. It fit the GPU at
  batch size 16, using about 7.4 of 7.5 GiB, and reached 0.137 MSE / 0.234 MAE after its
  60-minute cap. The paper reports 0.130 / 0.222. This beat the 32-channel run's 0.145 / 0.239
  and DLinear's 0.142 / 0.240. The full and subset runs also differed in training-window stride,
  batch size and epochs, so their difference is not a controlled estimate of channel count alone.
- I generated `results/comparison.md` from the recorded CSVs. It includes every completed main
  and extra result alongside the corresponding paper value and the run's deviations.
