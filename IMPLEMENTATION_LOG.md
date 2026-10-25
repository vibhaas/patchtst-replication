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
- Constraints: one RTX 5050 laptop GPU with 7.5 GiB usable memory. The recorded training took
  about 3.5 hours in total.
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
100), every 2nd training window on the larger datasets, mixed precision on Weather, Electricity and Traffic,
and shorter pre-training (20 epochs instead of 100).

## 2026-10-25: Main replication on the RTX 5050 laptop

- We installed PyTorch 2.11.0 with CUDA 12.8 in a local virtual environment. The RTX 5050 Laptop GPU
  reported 7.5 GiB of memory, and the CUDA kernel and autocast check passed. We downloaded the ETT,
  Weather, Electricity and Traffic CSVs from the configured mirror.
- The smoke plan exposed two setup problems that we fixed before the main runs. The source copy omitted
  the four official self-supervised data loader files; we restored them from the pinned upstream commit
  and applied the documented pandas 3 compatibility change. The unpatched, channel-independent model
  exceeded GPU memory at batch size 128, so we set its smoke and block B batch size to 4. We also used
  batch size 4 for the unpatched channel-mixing variant in block B. All seven smoke paths then passed.
- We ran all 53 main jobs in 100.8 minutes with no failures. The supervised jobs used `cuda:0`.
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
- We ran 20 epochs of ETTh1 masked pre-training, then linear probing and fine-tuning. At horizon 96,
  MSE was 0.368 for the probe and 0.370 for fine-tuning; the paper reports 0.371 and 0.366. At
  horizon 336, MSE was 0.445 and 0.428; the paper reports 0.445 and 0.450. These runs used the
  shortened schedules documented above.

## 2026-10-25: Remaining dataset and horizon coverage

- We completed the remaining horizon and dataset jobs on the same GPU. The 12 jobs in this part of
  the plan took about 107 minutes, including about 24 minutes of Traffic training. Along with the
  100.8-minute main plan, the total was about 208 minutes of training and evaluation.
- PatchTST/64 on ETTh1 and ETTh2 covered all four horizons for each dataset. All eight MSE values
  were within 1.3% of the paper. At horizon 96, ETTh1 reached 0.375 and ETTh2 reached 0.274.
- The added Weather horizons reached 0.201 MSE / 0.246 MAE at 192 and 0.320 / 0.335 at 720,
  compared with the paper's 0.197 / 0.243 and 0.320 / 0.335.
- We trained PatchTST on all 321 Electricity channels and all training windows. It fit the GPU at
  batch size 16, using about 7.4 of 7.5 GiB, and reached 0.137 MSE / 0.234 MAE after its
  60-minute cap. The paper reports 0.130 / 0.222. This beat the 32-channel run's 0.145 / 0.239
  and DLinear's 0.142 / 0.240. The full and subset runs also differed in training-window stride,
  batch size and epochs, so their difference is not a controlled estimate of channel count alone.
- Traffic/96 trained on 64 of 862 channels, with every second training window and mixed precision.
  We tested the shared model on all 862 channels. It reached 0.392 MSE / 0.271 MAE, compared with
  0.367 / 0.251 for the paper's full-channel PatchTST/42 setup. Training completed 30 epochs in
  about 24 minutes. Testing all channels at batch size 24 exceeded the 7.5 GiB GPU memory, so we
  loaded the saved best checkpoint and evaluated it with batch size 1. The predictions were scored
  batch by batch, without keeping the whole test set in RAM. This memory change did not retrain the
  model. The 6.9% MSE gap from the paper is plausible under our shorter, subset-channel schedule,
  but we cannot assign it to one setting alone.
- We generated `results/comparison.md` from the recorded CSVs. It includes every completed main
  and remaining result alongside the corresponding paper value and the run's deviations.

## 2026-10-25: Model path notes

We went back through the model code so the main steps are written down in one place. The forecast input is a rolling window with shape batch, past time steps, channels. The output has the same channels, but covers the future horizon. For example, the ETTh1 main run looks at 336 hourly points and predicts the next 96.

### PatchTST forecast path

- RevIN first normalizes each sample and each channel using that window's own statistics. It helps when one channel has a much larger scale than another. After the prediction, RevIN puts the output back on the original scale. The affine option was off in these runs.
- The window is split into overlapping patches. We set patch length to 16 and stride to 8, so adjacent patches share half their values. The code pads the end of the window. With a 336-step look-back this makes 42 patches. This is the sequence the Transformer processes, instead of all 336 individual time points.
- Each 16-value patch is projected to a feature vector, then given a position embedding. The position part lets the model know which patch came earlier or later. Three Transformer encoder layers use attention across the patches and update their features.
- Channel independence happens in the tensor reshape: batch and channel are folded together, so each channel is passed through the same encoder as a separate series. There is no attention from one variable to another. The prediction head is also shared in our config (`individual=0`); it flattens the encoded patches and maps them to the requested forecast length. The result is reshaped to batch, future steps, channels, then denormalized.

This sharing is why a model trained with 32 Electricity channels can still be applied to all 321 at test time. The same idea let us train on 64 Traffic channels and test on all 862. It is useful, but it does not by itself say a subset-channel run should score as well as training on every channel.

The plan uses a smaller model for hourly ETTh1 and ETTh2: three encoder layers, four heads, model width 16 and feed-forward width 128. ETTm, Weather, Electricity and Traffic use three layers, 16 heads, width 128 and feed-forward width 256. PatchTST batch size was lowered for memory on the larger datasets. The implementation is in `upstream/PatchTST_supervised/models/PatchTST.py` and `upstream/PatchTST_supervised/layers/PatchTST_backbone.py`.

### Baseline and ablation versions

DLinear is a short baseline to compare against. It uses a moving average with kernel 25 to split the input into a smoother trend and a residual part. Two linear maps predict those parts into the future; the predictions are added together. It has no attention layers.

The ablation runner changes one design choice at a time. CI-only uses patch length 1 and stride 1, so values are not grouped but each channel is still processed separately. P-only uses `PatchTST_CM`, our channel-mixing version: a time patch contains values from all channels and the Transformer can learn across them. The “Original” row uses that model with patch length 1, so it has neither patches nor channel independence. The mixing version was added for this project because the official code did not include it. So it is a useful check of the trend, though it is our implementation of the comparison.

### How the runs were managed

`scripts/run_plan.py` stores the jobs in priority order and starts them one at a time as subprocesses. Each job gets a stable tag. Results go into CSV rows and the logs go under `results/logs/`; completed tags are skipped on the next run, which made it possible to resume without repeating finished work. The runner stops starting jobs when the overall budget has passed. Inside a training job, Adam minimizes MSE, OneCycleLR changes the learning rate, validation loss controls early stopping, and a per-run time cap tests the best checkpoint reached so far.

The loader uses the authors' chronological train, validation and test split and forms rolling windows. `train_window_stride` keeps every k-th training window to shorten training; it does not thin the validation or test sets. For the Electricity and Traffic subset runs, a seeded random set of channels is used for train and validation, while the test path uses every channel. Mixed precision was enabled on Weather, Electricity and Traffic to fit the available memory and time.

### Masked pre-training path

The self-supervised experiment has its own upstream training scripts. It makes non-overlapping 12-value patches from a 512-step context and hides 40% of them. The encoder is trained to reconstruct the hidden patch values; the loss is measured only on masked patches. After pre-training, the linear probe freezes the encoder and trains a small forecast head. Fine-tuning first trains the head, then allows the full network to update. We used 20 pre-training epochs and shorter probe and fine-tuning schedules than the paper. The ETTh1/96 linear probe reached 0.368 MSE against 0.371 in the paper; that is close, but it is still one shortened run rather than a full reproduction of the paper's self-supervised study.
