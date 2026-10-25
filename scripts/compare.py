"""Compare results/*.csv with the paper's numbers and write results/comparison.md.

    python scripts/compare.py

Paper numbers come from reference/paper_results.csv (Tables 3, 9, 10 and 12 of the paper).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_plan import ROOT, build_plans  # noqa: E402

RESULTS = os.path.join(ROOT, 'results')


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def main():
    paper = {}
    for p in read_csv(os.path.join(ROOT, 'reference', 'paper_results.csv')):
        paper[(p['table'], p['data'], p['variant'], p['seq_len'], int(p['pred_len']))] = (float(p['mse']), float(p['mae']))

    ours = {}  # latest row per tag wins
    for row in read_csv(os.path.join(RESULTS, 'supervised.csv')) + read_csv(os.path.join(RESULTS, 'selfsup.csv')):
        ours[row['run_tag']] = row

    runs = {r.tag: r for plan in build_plans().values() for r in plan}

    def ref(table, data, variant, seq, h):
        return paper.get((table, data, variant, str(seq), h))

    lines = ['# PatchTST replication: results vs. paper', '',
             'Delta = (ours - paper) / paper on MSE. Negative means we did better. '
             'Notes list where a run deviates from the paper setup.', '',
             '| Block | Run | MSE | MAE | Paper MSE | Paper MAE | Delta MSE | Notes |',
             '|---|---|---|---|---|---|---|---|']
    for tag, row in ours.items():
        r = runs.get(tag)
        if r is None or tag.startswith('smoke/'):
            continue
        p = ref(r.paper[0], r.data, r.paper[1], r.paper[2], r.pred_len) if r.paper else None
        mse, mae = float(row['mse']), float(row['mae'])
        notes = []
        if row.get('train_channels') and row.get('enc_in') and row['train_channels'] != row['enc_in']:
            notes.append(f"trained on {row['train_channels']}/{row['enc_in']} channels")
        if row.get('train_window_stride', '1') not in ('', '1'):
            notes.append(f"training window stride {row['train_window_stride']}")
        if row.get('epochs'):
            notes.append(f"{row['epochs']} ep, {row.get('stopped_by', '')}, {row.get('train_minutes', '?')} min")
        if row.get('mode'):
            notes.append(row['mode'])
        pm, pa, delta = ('', '', '')
        if p:
            pm, pa, delta = f'{p[0]:.3f}', f'{p[1]:.3f}', f'{(mse - p[0]) / p[0] * 100:+.1f}%'
        lines.append(f'| {r.block} | {tag} | {mse:.3f} | {mae:.3f} | {pm} | {pa} | {delta} | {"; ".join(notes)} |')

    # Ablation pivot (block B, plus the full model from block A)
    lines += ['', '## Ablation on ETTh1 (paper Table 10): MSE ours / paper', '',
              '| Horizon | P+CI (full) | CI only | P only | Original |', '|---|---|---|---|---|']
    for h in [96, 336]:
        cells = []
        for tag, variant in [(f'A/PatchTST42/ETTh1/{h}', 'P+CI'), (f'B/CI-only/ETTh1/{h}', 'CI'),
                             (f'B/P-only/ETTh1/{h}', 'P'), (f'B/Original/ETTh1/{h}', 'Original')]:
            p = ref('table10', 'ETTh1', variant, '', h)
            o = f"{float(ours[tag]['mse']):.3f}" if tag in ours else '-'
            cells.append(f"{o} / {p[0]:.3f}" if p else o)
        lines.append(f'| {h} | ' + ' | '.join(cells) + ' |')

    # Look-back pivot (block C, plus L=336 from block A)
    lines += ['', '## Look-back window on ETTh1, horizon 96 (paper Table 9): MSE ours / paper', '',
              '| L | ours | paper |', '|---|---|---|']
    for L in [96, 192, 336, 512, 720]:
        tag = 'A/PatchTST42/ETTh1/96' if L == 336 else f'C/PatchTST/ETTh1/96/L{L}'
        p = ref('table9', 'ETTh1', 'PatchTST/42', L, 96)
        o = f"{float(ours[tag]['mse']):.3f}" if tag in ours else '-'
        lines.append(f"| {L} | {o} | {f'{p[0]:.3f}' if p else 'n/a'} |")

    out = os.path.join(RESULTS, 'comparison.md')
    os.makedirs(RESULTS, exist_ok=True)
    with open(out, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nwritten to {os.path.relpath(out, ROOT)}')


if __name__ == '__main__':
    main()
