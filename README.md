# PatchTST replication

College project: replicating *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*
(Nie et al., ICLR 2023, [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)) with the authors' code, on a
single 6 GB laptop GPU and a ~3 hour training budget.

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx; pick the right build for other GPUs
pip install -r requirements.txt
python scripts/check_env.py
python scripts/download_data.py
python scripts/run_plan.py --plan main
python scripts/compare.py        # -> results/comparison.md
```

## Contents

- `paper/`: the paper
- `upstream/`: official code ([yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST), Apache-2.0), changes listed in [`UPSTREAM_PATCHES.md`](UPSTREAM_PATCHES.md)
- `scripts/`: environment check, data download, experiment runner, comparison with the paper
- `reference/paper_results.csv`: the paper's result tables
- `results/`: our results and the comparison
- [`IMPLEMENTATION_LOG.md`](IMPLEMENTATION_LOG.md): project log
