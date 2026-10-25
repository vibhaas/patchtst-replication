"""Budgeted experiment runner for the PatchTST replication.

Runs the official code in upstream/ as subprocesses, one run at a time, in priority order.
Every finished run appends a row to results/supervised.csv or results/selfsup.csv, and runs that
already have a row are skipped, so the plan can be stopped and resumed at any point.

    python scripts/run_plan.py --plan smoke            # ~5 min sanity check (CPU is fine)
    python scripts/run_plan.py --plan main --dry-run   # list the main plan with time estimates
    python scripts/run_plan.py --plan main             # the ~3 h plan (GPU)
    python scripts/run_plan.py --plan main --blocks A,B --budget-minutes 60

See HANDOFF.md for what each block replicates.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUP_DIR = os.path.join(ROOT, 'upstream', 'PatchTST_supervised')
SS_DIR = os.path.join(ROOT, 'upstream', 'PatchTST_self_supervised')

# ---------------------------------------------------------------------------
# Per-dataset settings, copied from upstream/PatchTST_supervised/scripts/{PatchTST,Linear}/*.sh
# ---------------------------------------------------------------------------
DATA = {  # name -> (csv file, --data value for the loader, number of channels)
    'ETTh1': ('ETTh1.csv', 'ETTh1', 7),
    'ETTh2': ('ETTh2.csv', 'ETTh2', 7),
    'ETTm1': ('ETTm1.csv', 'ETTm1', 7),
    'ETTm2': ('ETTm2.csv', 'ETTm2', 7),
    'weather': ('weather.csv', 'custom', 21),
    'electricity': ('electricity.csv', 'custom', 321),
    'traffic': ('traffic.csv', 'custom', 862),
}
_SMALL = dict(e_layers=3, n_heads=4, d_model=16, d_ff=128, dropout=0.3, fc_dropout=0.3, head_dropout=0,
              batch_size=128, learning_rate=0.0001, lradj='type3', pct_start=0.3)
_BIG = dict(e_layers=3, n_heads=16, d_model=128, d_ff=256, dropout=0.2, fc_dropout=0.2, head_dropout=0,
            batch_size=128, learning_rate=0.0001)
PATCHTST = {
    'ETTh1': _SMALL, 'ETTh2': _SMALL,
    'ETTm1': {**_BIG, 'lradj': 'TST', 'pct_start': 0.4},
    'ETTm2': {**_BIG, 'lradj': 'TST', 'pct_start': 0.4},
    'weather': {**_BIG, 'lradj': 'type3', 'pct_start': 0.3},
    'electricity': {**_BIG, 'batch_size': 32, 'lradj': 'TST', 'pct_start': 0.2},
    'traffic': {**_BIG, 'batch_size': 24, 'lradj': 'TST', 'pct_start': 0.2},
}
DLINEAR = {  # (batch_size, learning_rate); ETTm2 uses a different lr per horizon upstream
    'ETTh1': (32, 0.005), 'ETTh2': (32, 0.05), 'ETTm1': (8, 0.0001),
    'ETTm2': (32, {96: 0.001, 192: 0.001, 336: 0.01, 720: 0.1}),
    'weather': (16, 0.0001), 'electricity': (16, 0.001), 'traffic': (16, 0.05),
}

# Budget defaults (deviations from the paper, see HANDOFF.md "Deviations")
PATCHTST_EPOCHS, PATCHTST_PATIENCE = 30, 5       # paper: 100 epochs, patience 20 (or none)
DLINEAR_EPOCHS, DLINEAR_PATIENCE = 10, 3         # LTSF-Linear defaults


@dataclass
class Run:
    tag: str            # unique id, also the resume key
    block: str          # plan block letter
    kind: str           # 'sup' | 'pretrain' | 'linear_probe' | 'finetune'
    data: str
    pred_len: int = 96
    model: str = 'PatchTST'
    seq_len: int = 336
    est_min: float = 5  # rough GPU estimate, only used for --dry-run totals
    max_minutes: float = 10
    paper: tuple = ()   # (table, variant, seq_len or '') in reference/paper_results.csv
    args: dict = field(default_factory=dict)


def sup(tag, block, data, pred_len, model='PatchTST', seq_len=336, est=5, cap=10, paper=(), **over):
    return Run(tag, block, 'sup', data, pred_len, model, seq_len, est, cap, paper, over)


def build_plans():
    H = [96, 192, 336, 720]
    plans = {}

    # --- smoke: proves every code path runs; numbers are meaningless -----------------------
    q = dict(train_epochs=1, train_window_stride=20)
    plans['smoke'] = [
        sup('smoke/PatchTST/ETTh1/96', 'S', 'ETTh1', 96, est=1, cap=2, **q),
        sup('smoke/DLinear/ETTh1/96', 'S', 'ETTh1', 96, model='DLinear', est=1, cap=2, **q),
        sup('smoke/P-only/ETTh1/96', 'S', 'ETTh1', 96, model='PatchTST_CM', est=1, cap=2, **q),
        sup('smoke/CI-only/ETTh1/96', 'S', 'ETTh1', 96, est=1, cap=2, patch_len=1, stride=1, batch_size=4, **{**q, 'train_window_stride': 40}),
        sup('smoke/subset3/ETTh1/96', 'S', 'ETTh1', 96, est=1, cap=2, train_channels=3, **q),
        Run('smoke/ss-pretrain/ETTh1', 'S', 'pretrain', 'ETTh1', est_min=2, args=dict(n_epochs_pretrain=1)),
        Run('smoke/ss-linprobe/ETTh1/96', 'S', 'linear_probe', 'ETTh1', 96, est_min=1, args=dict(n_epochs_pretrain=1, n_epochs_finetune=1)),
    ]

    main = []
    # --- A: headline, Table 3 (PatchTST/42 vs DLinear on the four ETT sets) ----------------
    for d in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        big = d.startswith('ETTm')
        for h in H:
            main.append(sup(f'A/PatchTST42/{d}/{h}', 'A', d, h, est=4 if big else 2, cap=6 if big else 4,
                            paper=('table3', 'PatchTST/42', ''), **({'train_window_stride': 2} if big else {})))
            main.append(sup(f'A/DLinear/{d}/{h}', 'A', d, h, model='DLinear', est=1.5 if big else 0.5, cap=4,
                            paper=('table3', 'DLinear', '')))
    # --- B: ablation, Table 7/10 (patching x channel-independence) on ETTh1 ------------------
    for h in [96, 336]:
        main.append(sup(f'B/CI-only/ETTh1/{h}', 'B', 'ETTh1', h, est=5, cap=8, patch_len=1, stride=1, batch_size=4,
                        paper=('table10', 'CI', '')))
        main.append(sup(f'B/P-only/ETTh1/{h}', 'B', 'ETTh1', h, model='PatchTST_CM', est=2, cap=5,
                        paper=('table10', 'P', '')))
        main.append(sup(f'B/Original/ETTh1/{h}', 'B', 'ETTh1', h, model='PatchTST_CM', est=4, cap=8, patch_len=1, stride=1, batch_size=4,
                        paper=('table10', 'Original', '')))
    # --- C: look-back window, Figure 2 / Table 9, ETTh1 horizon 96 (L=336 comes from block A) --
    for L in [96, 192, 512, 720]:
        main.append(sup(f'C/PatchTST/ETTh1/96/L{L}', 'C', 'ETTh1', 96, seq_len=L, est=2, cap=4,
                        paper=('table9', 'PatchTST/42', L)))
    # --- D: a larger dataset, Weather (21 channels, full-size model) ---------------------------
    for h in [96, 336]:
        main.append(sup(f'D/PatchTST42/weather/{h}', 'D', 'weather', h, est=12, cap=15, train_window_stride=2, use_amp=True,
                        paper=('table3', 'PatchTST/42', '')))
        main.append(sup(f'D/DLinear/weather/{h}', 'D', 'weather', h, model='DLinear', est=2, cap=5,
                        paper=('table3', 'DLinear', '')))
    # --- E: own experiment, Electricity trained on 32 of 321 channels, tested on all 321 ------
    main.append(sup('E/PatchTST42-32ch/electricity/96', 'E', 'electricity', 96, est=20, cap=25, train_channels=32,
                    train_window_stride=2, use_amp=True, paper=('table3', 'PatchTST/42', '')))
    main.append(sup('E/DLinear/electricity/96', 'E', 'electricity', 96, model='DLinear', est=5, cap=8,
                    paper=('table3', 'DLinear', '')))
    # --- F: self-supervised, Table 12 (pre-train on ETTh1, then linear probe / fine-tune) ------
    ss = dict(n_epochs_pretrain=20, n_epochs_finetune=10, freeze_epochs=5)
    main += [
        Run('F/ss-pretrain/ETTh1', 'F', 'pretrain', 'ETTh1', est_min=6, args=ss),
        Run('F/ss-linprobe/ETTh1/96', 'F', 'linear_probe', 'ETTh1', 96, est_min=3, paper=('table12', 'SS-linear_probe', ''), args=ss),
        Run('F/ss-finetune/ETTh1/96', 'F', 'finetune', 'ETTh1', 96, est_min=5, paper=('table12', 'SS-finetune', ''), args=ss),
        Run('F/ss-linprobe/ETTh1/336', 'F', 'linear_probe', 'ETTh1', 336, est_min=3, paper=('table12', 'SS-linear_probe', ''), args=ss),
        Run('F/ss-finetune/ETTh1/336', 'F', 'finetune', 'ETTh1', 336, est_min=5, paper=('table12', 'SS-finetune', ''), args=ss),
    ]
    plans['main'] = main

    # --- extra: only if time is left over --------------------------------------------------------
    extra = []
    for d in ['ETTh1', 'ETTh2']:
        for h in H:
            extra.append(sup(f'X/PatchTST64/{d}/{h}', 'X', d, h, seq_len=512, est=2, cap=4, paper=('table3', 'PatchTST/64', '')))
    for h in [192, 720]:
        extra.append(sup(f'X/PatchTST42/weather/{h}', 'X', 'weather', h, est=12, cap=15, train_window_stride=2, use_amp=True,
                         paper=('table3', 'PatchTST/42', '')))
    extra.append(sup('X/PatchTST42-full/electricity/96', 'X', 'electricity', 96, est=60, cap=60, batch_size=16, use_amp=True,
                     paper=('table3', 'PatchTST/42', '')))
    extra.append(sup('X/PatchTST42-64ch/traffic/96', 'X', 'traffic', 96, est=40, cap=40, train_channels=64, batch_size=24,
                     train_window_stride=2, use_amp=True, paper=('table3', 'PatchTST/42', '')))
    plans['extra'] = extra
    plans['all'] = main + extra
    return plans


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------
def sup_cmd(r, opts):
    csv_file, loader, n_ch = DATA[r.data]
    a = dict(random_seed=2021, is_training=1, root_path=opts.data_dir + os.sep, data_path=csv_file,
             model_id=f'{r.data}_{r.seq_len}_{r.pred_len}', model=r.model, data=loader, features='M',
             seq_len=r.seq_len, pred_len=r.pred_len, enc_in=n_ch, des='Exp', itr=1,
             num_workers=opts.num_workers, epoch_test=0, max_minutes=r.max_minutes,
             results_csv=os.path.join(opts.results_dir, 'supervised.csv'), run_tag=r.tag)
    if r.model == 'DLinear':
        bs, lr = DLINEAR[r.data]
        a.update(batch_size=bs, learning_rate=lr[r.pred_len] if isinstance(lr, dict) else lr,
                 train_epochs=DLINEAR_EPOCHS, patience=DLINEAR_PATIENCE)
    else:
        a.update(PATCHTST[r.data], patch_len=16, stride=8, train_epochs=PATCHTST_EPOCHS, patience=PATCHTST_PATIENCE)
    a.update(r.args)
    if opts.amp_off:
        a.pop('use_amp', None)
    cmd = [sys.executable, '-u', 'run_longExp.py']
    for k, v in a.items():
        if v is True:
            cmd.append(f'--{k}')
        elif v is not False:
            cmd += [f'--{k}', str(v)]
    cmd += ['--no_plots'] + (['--force_cpu'] if opts.device == 'cpu' else [])
    return SUP_DIR, cmd


SS_MODEL = dict(context_points=512, patch_len=12, stride=12, n_layers=3, n_heads=16, d_model=128, d_ff=256,
                dropout=0.2, head_dropout=0.2, revin=1, batch_size=64, num_workers=0)


def pretrained_path(r):
    e = r.args.get('n_epochs_pretrain', 10)
    return os.path.join(SS_DIR, 'saved_models', r.data.lower(), 'masked_patchtst', 'based_model',
                        f'patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain{e}_mask0.4_model1.pth')


def ss_cmd(r, opts):
    a = dict(SS_MODEL)
    if r.kind == 'pretrain':
        a.update(dset_pretrain=r.data.lower(), target_points=96, mask_ratio=0.4,
                 n_epochs_pretrain=r.args.get('n_epochs_pretrain', 10))
        script = 'patchtst_pretrain.py'
    else:
        a.update(dset_finetune=r.data.lower(), target_points=r.pred_len, pretrained_model=pretrained_path(r),
                 n_epochs_finetune=r.args.get('n_epochs_finetune', 10), freeze_epochs=r.args.get('freeze_epochs', 10),
                 results_csv=os.path.join(opts.results_dir, 'selfsup.csv'), run_tag=r.tag,
                 **{'is_linear_probe' if r.kind == 'linear_probe' else 'is_finetune': 1})
        script = 'patchtst_finetune.py'
    cmd = [sys.executable, '-u', script]
    for k, v in a.items():
        cmd += [f'--{k}', str(v)]
    return SS_DIR, cmd


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def done_tags(results_dir):
    tags = set()
    for name in ['supervised.csv', 'selfsup.csv']:
        p = os.path.join(results_dir, name)
        if os.path.exists(p):
            with open(p, newline='') as fh:
                tags |= {row['run_tag'] for row in csv.DictReader(fh)}
    return tags


def is_done(r, opts, finished):
    if r.kind == 'pretrain':
        return os.path.exists(pretrained_path(r))
    return r.tag in finished


def main():
    plans = build_plans()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', choices=sorted(plans), default='smoke')
    ap.add_argument('--blocks', default='', help='comma-separated block letters to keep, e.g. A,B')
    ap.add_argument('--budget-minutes', type=float, default=175, help='stop starting new runs after this much wall time')
    ap.add_argument('--device', choices=['auto', 'cpu'], default='auto')
    ap.add_argument('--amp-off', action='store_true', help='disable mixed precision even where the plan enables it')
    ap.add_argument('--data-dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--results-dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--num-workers', type=int, default=0 if os.name == 'nt' else 2)
    ap.add_argument('--rerun', action='store_true', help='ignore existing results and run everything again')
    ap.add_argument('--dry-run', action='store_true', help='print the runs and commands without executing')
    opts = ap.parse_args()
    opts.data_dir, opts.results_dir = os.path.abspath(opts.data_dir), os.path.abspath(opts.results_dir)

    runs = plans[opts.plan]
    if opts.blocks:
        keep = set(opts.blocks.upper().split(','))
        runs = [r for r in runs if r.block in keep]
    finished = set() if opts.rerun else done_tags(opts.results_dir)
    todo = [r for r in runs if opts.rerun or not is_done(r, opts, finished)]

    needed = {DATA[r.data][0] for r in todo}
    missing = sorted(f for f in needed if not os.path.exists(os.path.join(opts.data_dir, f)))
    print(f'plan={opts.plan}: {len(runs)} runs, {len(runs) - len(todo)} already done, {len(todo)} to go, '
          f'~{sum(r.est_min for r in todo):.0f} min estimated (GPU), budget {opts.budget_minutes:.0f} min')
    if missing:
        print('missing data files:', ', '.join(missing), '-> run: python scripts/download_data.py')
        if not opts.dry_run:
            sys.exit(1)

    os.makedirs(os.path.join(opts.results_dir, 'logs'), exist_ok=True)
    env = dict(os.environ, PATCHTST_DATA=opts.data_dir, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    if opts.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    start = time.time()
    failures = []
    for i, r in enumerate(todo, 1):
        cwd, cmd = sup_cmd(r, opts) if r.kind == 'sup' else ss_cmd(r, opts)
        if opts.dry_run:
            print(f'[{r.block}] {r.tag:40s} ~{r.est_min:>4.1f} min  cap {r.max_minutes:>4.0f}')
            print('      cd', os.path.relpath(cwd, ROOT), '&&', ' '.join(cmd[1:]))
            continue
        elapsed = (time.time() - start) / 60
        if elapsed > opts.budget_minutes:
            print(f'budget of {opts.budget_minutes:.0f} min used up, stopping before {r.tag} ({len(todo) - i + 1} runs left)')
            break
        if r.kind in ('linear_probe', 'finetune') and not os.path.exists(pretrained_path(r)):
            print(f'[{i}/{len(todo)}] {r.tag}  SKIPPED: no pre-trained model (did the pretrain run fail?)', flush=True)
            failures.append(r.tag)
            continue
        log = os.path.join(opts.results_dir, 'logs', r.tag.replace('/', '__') + '.log')
        print(f'[{i}/{len(todo)}] {r.tag}  (elapsed {elapsed:.1f} min)  log: {os.path.relpath(log, ROOT)}', flush=True)
        t0 = time.time()
        with open(log, 'w') as fh:
            fh.write(' '.join(cmd) + '\n\n')
            fh.flush()
            rc = subprocess.call(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        status = 'ok' if rc == 0 else f'FAILED (exit {rc}, see log)'
        print(f'      {status} in {(time.time() - t0) / 60:.1f} min', flush=True)
        if rc != 0:
            failures.append(r.tag)
    if not opts.dry_run:
        print(f'finished in {(time.time() - start) / 60:.1f} min; failures: {failures or "none"}')
        print('compare with the paper: python scripts/compare.py')
        sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
