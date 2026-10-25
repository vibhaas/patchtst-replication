# Changes to the official code

`upstream/` is a copy of https://github.com/yuqinie98/PatchTST at commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9` (Apache-2.0, see `upstream/LICENSE`).
`PatchTST_supervised/Formers/` (the FEDformer/Pyraformer baselines) and the README images were left out.
Every change is marked with a `# [replication]` comment.

The four `PatchTST_self_supervised/src/data/` files were omitted from this repo's initial copy.
They have been restored from the pinned official commit; `pred_dataset.py` has the pandas 3 fix below.

## Compatibility with current library versions

| File | Change |
|---|---|
| `PatchTST_supervised/utils/tools.py` | `np.Inf` -> `np.inf` (removed in NumPy 2) |
| `PatchTST_supervised/data_provider/data_loader.py`, `PatchTST_self_supervised/src/data/pred_dataset.py` | `Series.apply(f, 1)` -> `Series.apply(f)` (pandas 3 reads the 1 as `args`) |
| `PatchTST_self_supervised/src/callback/scheduler.py` | dropped `verbose=` from `OneCycleLR` (removed in recent PyTorch) |
| `PatchTST_supervised/exp/exp_main.py` | `torch.load(..., map_location=device)` so checkpoints load on CPU too |

## Running on one laptop GPU

| File | Change |
|---|---|
| `PatchTST_self_supervised/datautils.py` | hard-coded `/data/datasets/public/...` paths replaced by `PATCHTST_DATA` (default `<repo>/data`) |
| `PatchTST_self_supervised/src/basics.py` | `set_device()` works without CUDA and falls back to GPU 0 if all GPUs look busy (a laptop GPU driving the display) |
| `PatchTST_self_supervised/patchtst_finetune.py` | test step used a hard-coded `'cuda'`; now uses the default device |

## Budget and subsetting options (supervised, `run_longExp.py`)

| Option | Effect |
|---|---|
| `--train_channels K`, `--channel_seed` | train and validate on K random channels; test always on all channels. Works because PatchTST shares weights across channels (paper, Appendix A.1.5) |
| `--train_window_stride k` | use every k-th training window; validation and test windows are unchanged |
| `--max_minutes m` | stop training after m minutes, then test the best checkpoint |
| `--epoch_test 0` | skip the test-set evaluation after every epoch (upstream does it only for logging) |
| `--no_plots` | skip the PDF plots written during testing |
| `--test_batch_size` | use a smaller test batch when a subset-channel model is applied to all channels |
| `--stream_test_metrics` | calculate test MSE and MAE as batches arrive, avoiding large prediction arrays for Traffic |
| `--results_csv`, `--run_tag` | append one row per run (metrics, epochs, stop reason, time) to a CSV |
| `--force_cpu` | ignore the GPU |

## Additions

- `PatchTST_supervised/models/PatchTST_CM.py`: channel-mixing PatchTST for the ablation. Each token holds one
  patch of every channel (`B x (M*P) x N`, Appendix A.4.3). With `patch_len = stride = 1` it is the "Original"
  variant (no patching, channel-mixing). The official repo does not include this.
- `patchtst_finetune.py`: `--freeze_epochs` (head-only epochs before full fine-tuning, upstream fixed at 10),
  plus `--results_csv` and `--run_tag`.
