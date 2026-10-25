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

## Runs

*(to be filled in)*
