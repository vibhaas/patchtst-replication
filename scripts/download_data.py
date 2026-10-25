"""Download the benchmark CSVs into data/ (flat), from the Autoformer authors' Hugging Face mirror.

    python scripts/download_data.py              # what the main plan needs (~124 MB)
    python scripts/download_data.py --set ett    # only the four ETT files (~25 MB), enough for the smoke test
    python scripts/download_data.py --set all    # everything incl. traffic and illness (~265 MB)
    python scripts/download_data.py ETTh1 weather

Source: https://huggingface.co/datasets/thuml/Time-Series-Library (same files as the Google Drive
folder linked in the PatchTST README). Files that already exist with the right size are skipped.
"""
import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 'https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/'
FILES = {  # name -> (path in the mirror, size in bytes)
    'ETTh1': ('ETT-small/ETTh1.csv', 2589657),
    'ETTh2': ('ETT-small/ETTh2.csv', 2417960),
    'ETTm1': ('ETT-small/ETTm1.csv', 10360719),
    'ETTm2': ('ETT-small/ETTm2.csv', 9677236),
    'weather': ('weather/weather.csv', 7235425),
    'electricity': ('electricity/electricity.csv', 95581762),
    'traffic': ('traffic/traffic.csv', 136478119),
    'national_illness': ('illness/national_illness.csv', 67620),
}
SETS = {
    'ett': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'],
    'main': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity'],
    'all': list(FILES),
}


def fetch(name, out_dir):
    path, size = FILES[name]
    dest = os.path.join(out_dir, os.path.basename(path))
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f'{name:17s} ok (already there)')
        return
    tmp = dest + '.part'
    print(f'{name:17s} downloading {size / 1e6:.1f} MB ...', flush=True)
    urllib.request.urlretrieve(BASE + path, tmp)
    got = os.path.getsize(tmp)
    if got != size:
        os.remove(tmp)
        sys.exit(f'{name}: expected {size} bytes, got {got}; the mirror may have changed')
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='*', help=f'datasets: {", ".join(FILES)}')
    ap.add_argument('--set', choices=sorted(SETS), default='main')
    ap.add_argument('--out', default=os.path.join(ROOT, 'data'))
    a = ap.parse_args()
    names = a.names or SETS[a.set]
    unknown = [n for n in names if n not in FILES]
    if unknown:
        sys.exit(f'unknown dataset(s): {unknown}')
    os.makedirs(a.out, exist_ok=True)
    for n in names:
        fetch(n, a.out)
    print(f'done -> {a.out}')


if __name__ == '__main__':
    main()
