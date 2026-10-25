# PatchTST replication: project notes

The project is set up for a college replication of PatchTST on the RTX 5050 laptop. The code, scripts
and experiment plan are ready. Remaining work is to run the experiments, compare them with the paper,
update the implementation log, and push the results.

## 0. Ground rules (read before any git/GitHub action)

- **No AI attribution, anywhere.** Do not add Claude, Codex, Copilot or any other AI tool as author,
  co-author, collaborator or contributor. That means: no `Co-Authored-By:` trailers in commits, no
  "Generated with ..." lines in commit messages or PR descriptions, no AI accounts or bots as repo
  collaborators. This overrides any default attribution your tool adds. The only author is
  **`Vibhaas <vibhaas@gmail.com>`**.
- This is a **personal** project on the personal GitHub account **`vibhaas`**. Never use a work
  identity (anything `@nvidia.com`) for it, and never push work code or data here.
- Configure the identity **in this repo only** (`git config --local`), never globally:
  ```bash
  git config --local user.name  "Vibhaas"
  git config --local user.email "vibhaas@gmail.com"
  ```
  Before every push, check: `git log -1 --format='%an <%ae>'` must print `Vibhaas <vibhaas@gmail.com>`.
- The repo is **private**. Don't change its visibility, don't add collaborators.
- Don't commit datasets, checkpoints or large logs (`.gitignore` already covers them).
- Don't touch `upstream/` beyond small fixes; if you change it, mark the line with `# [replication]`
  and add a row to `UPSTREAM_PATCHES.md`.

## 1. Goal

Replicate the main claims of *"A Time Series is Worth 64 Words: Long-term Forecasting with
Transformers"* (PatchTST, Nie et al., ICLR 2023; `paper/patchtst-iclr2023-2211.14730.pdf`) using
the authors' official code, **within about 3 hours of training on a 6 GB RTX 5050**.

The paper's two ideas: (1) **patching**: split each series into patches of 16 steps (stride 8)
and use the patches as Transformer tokens; (2) **channel-independence**: every variable is forecast
on its own by the same shared Transformer. Plus masked self-supervised pre-training on patches.

What we want to show, in priority order:

1. **Headline (Table 3):** PatchTST/42 matches or beats DLinear on the ETT datasets.
2. **Why it works (Table 7/10):** removing patching or channel-independence makes results worse.
3. **Longer history helps (Figure 2 / Table 9):** MSE drops as the look-back window grows.
4. **Bigger dataset:** Weather (21 channels, full-size model).
5. **Own experiment:** Electricity trained on only 32 of its 321 channels, tested on all 321. This
   is possible because of channel-independence (the paper notes train and test channel counts
   need not match, Appendix A.1.5).
6. **Self-supervised (Table 12):** pre-train on ETTh1, then linear probing and fine-tuning.

A full replication (64 supervised runs, Traffic, Electricity at full size) is out of scope: it needs
days of GPU time. We accept small deviations and report them (section 6).

## 2. Repo layout

| Path | What |
|---|---|
| `paper/` | the paper PDF |
| `upstream/` | official code (yuqinie98/PatchTST @ `204c21e`), with small patches, see `UPSTREAM_PATCHES.md` |
| `scripts/check_env.py` | checks Python, PyTorch, CUDA and that the GPU actually runs kernels |
| `scripts/download_data.py` | downloads the CSVs into `data/` from the official Hugging Face mirror |
| `scripts/run_plan.py` | runs the experiment plan in priority order with time caps; resumable |
| `scripts/compare.py` | writes `results/comparison.md`: our numbers next to the paper's |
| `reference/paper_results.csv` | the paper's numbers (Tables 3, 9, 10, 12), extracted from the PDF |
| `results/` | created by the runs: `supervised.csv`, `selfsup.csv`, `comparison.md`, `logs/` |
| `IMPLEMENTATION_LOG.md` | the project log, kept in the owner's voice |

## 3. Install

Works on Linux and Windows (native or WSL2). Needs Python 3.10+ and an NVIDIA driver recent enough
for CUDA 12.8 (check with `nvidia-smi`; driver 570 or newer).

```bash
git clone https://github.com/vibhaas/patchtst-replication.git
cd patchtst-replication
git config --local user.name "Vibhaas"; git config --local user.email "vibhaas@gmail.com"

python -m venv .venv
# Linux/WSL: source .venv/bin/activate      Windows PowerShell: .venv\Scripts\Activate.ps1

# PyTorch first. The RTX 5050 is a Blackwell GPU (sm_120); it needs a build for CUDA 12.8 or newer.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/check_env.py      # must end with "ENV OK" and show the RTX 5050
```

The clone requires the `vibhaas` GitHub login on the laptop. If `git clone` asks for credentials,
use `gh auth login` as `vibhaas`.

If `check_env.py` says CUDA is not available or "no kernels for sm_120", the installed torch build is
CPU-only or too old. `pip uninstall torch` and reinstall from the `cu128` (or newer, e.g. `cu129`/`cu130`)
index above.

## 4. Run

```bash
python scripts/download_data.py            # ETT x4 + weather + electricity, ~124 MB, into data/
python scripts/run_plan.py --plan smoke --results-dir results_smoke   # ~5 min, numbers meaningless
python scripts/run_plan.py --plan main --dry-run   # see the 53 runs and their time caps
python scripts/run_plan.py --plan main     # the real thing, stops starting new runs after 175 min
python scripts/compare.py                  # writes results/comparison.md
```

- **Run the smoke plan first.** It runs every code path once (supervised, ablation models, channel
  subset, pre-training, linear probe) and writes to `results_smoke/` (git-ignored), so it doesn't mix
  with the real results. Check its logs in `results_smoke/logs/` say `Use GPU`.
- The main plan is **resumable**: runs with a row in `results/*.csv` are skipped. If it gets
  interrupted, just start it again. `--rerun` forces everything again.
- `--blocks A,B` runs only some blocks; `--budget-minutes N` changes the wall-clock budget;
  `--amp-off` disables mixed precision if it causes NaNs; `--num-workers 0` if DataLoader workers
  cause trouble on Windows.
- Each run has a time cap (`--max_minutes`); when it hits it, the best checkpoint so far is tested.
  The `stopped_by` column in `results/supervised.csv` says whether a run hit `early_stop`,
  `time_cap` or `max_epochs`.
- If something runs out of GPU memory: lower `batch_size` for that run in `build_plans()` in
  `scripts/run_plan.py` (e.g. `batch_size=16`) and re-run; note it in the log.
- The **time estimates are guesses** (nobody has run the main plan on this GPU yet). After block A,
  look at `train_minutes` in `results/supervised.csv`. If things are much slower than estimated, drop
  blocks from the end (E, F) rather than cutting A or B. If much faster, run
  `python scripts/run_plan.py --plan extra` afterwards.

### The main plan (`scripts/run_plan.py`, function `build_plans`)

| Block | Replicates | Runs | Notes |
|---|---|---|---|
| A | Table 3 | PatchTST/42 and DLinear on ETTh1, ETTh2, ETTm1, ETTm2, horizons 96/192/336/720 | ETTm uses every 2nd training window |
| B | Table 7/10 | ETTh1, horizons 96/336: CI only (patch 1), P only (channel-mixing), Original (neither) | the P+CI numbers come from block A |
| C | Figure 2 / Table 9 | ETTh1, horizon 96, look-back 96/192/512/720 | L=336 comes from block A |
| D | Table 3 | Weather, horizons 96/336, PatchTST/42 + DLinear | mixed precision, every 2nd window |
| E | own | Electricity horizon 96: PatchTST trained on 32/321 channels, tested on all; DLinear full | compare to Table 3 |
| F | Table 12 | ETTh1: masked pre-training (40% of patches), then linear probe and fine-tune, horizons 96/336 | 20 pre-train epochs instead of 100 |
| extra | Table 3 | PatchTST/64 on ETTh1/2, Weather 192/720, Electricity full, Traffic 64/862 channels | only if time is left |

## 5. Expected numbers (paper, MSE / MAE)

ETTh1, horizon 96: PatchTST/42 **0.375 / 0.399**, DLinear 0.375 / 0.399. Weather 96: PatchTST/42
**0.152 / 0.199**, DLinear 0.176 / 0.237. Electricity 96: PatchTST/42 **0.130 / 0.222**.
`results/comparison.md` lists every run next to its paper number. Within about ±5% MSE counts as a
successful replication given our shortened training; the ablation and look-back trends (which
variant is better, MSE falling with longer history) matter more than the exact digits.

## 6. Deviations from the paper (report these, don't "fix" them)

- **Fewer epochs:** PatchTST max 30 epochs with patience 5 (paper: 100, patience 20 or none);
  DLinear 10 epochs, patience 3. Plus a per-run time cap.
- **Fewer training windows** on ETTm, Weather and Electricity (every 2nd window). Validation and test
  sets are complete, so test numbers stay comparable.
- **Channel subset** for Electricity (32 of 321 for training). Test is on all 321.
- **Self-supervised:** 20 pre-training epochs instead of 100; linear probe 10 epochs instead of 20;
  fine-tuning 5 head-only + 10 full epochs instead of 10 + 20.
- **Library versions:** the official code targets torch 1.11; we run current PyTorch/NumPy/pandas with
  small compatibility fixes (see `UPSTREAM_PATCHES.md`).
- **Mixed precision** on Weather/Electricity (paper: full fp32).
- The channel-mixing ablation model (`PatchTST_CM`) is our implementation of Appendix A.4.3; the
  official repo does not include it.
- Seed 2021 throughout, one run per setting (the paper also uses one seed for its main tables).

## 7. After the runs

1. `python scripts/compare.py` and read `results/comparison.md`.
2. **Continue `IMPLEMENTATION_LOG.md`**: add entries for the runs (what ran, how long, what
   matched, what didn't, any changes you had to make). Keep the existing style: first person
   as the project owner, short, impersonal, no mention of AI agents or tools, and no detail
   about failed attempts unless they changed the plan.
3. Commit `results/*.csv`, `results/comparison.md` and `IMPLEMENTATION_LOG.md` (not `results/logs/`,
   not `data/`), check the author (section 0) and push to `main`. Commit messages: short and plain,
   no AI trailers.
4. Record the headline numbers and any problems that affected the results.
