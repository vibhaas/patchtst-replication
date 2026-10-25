"""Check that Python, PyTorch and the GPU are usable before spending time on training.

    python scripts/check_env.py
"""
import importlib
import platform
import sys


def main():
    ok = True
    print(f'python   {sys.version.split()[0]} on {platform.system()} {platform.machine()}')
    if sys.version_info < (3, 9):
        print('  !! need Python 3.9 or newer'); ok = False
    for mod in ['numpy', 'pandas', 'sklearn', 'matplotlib']:
        try:
            m = importlib.import_module(mod)
            print(f'{mod:8s} {getattr(m, "__version__", "?")}')
        except ImportError:
            print(f'{mod:8s} MISSING -> pip install -r requirements.txt'); ok = False
    try:
        import torch
    except ImportError:
        print('torch    MISSING -> see HANDOFF.md "Install"'); sys.exit(1)
    print(f'torch    {torch.__version__} (built for CUDA {torch.version.cuda})')
    if not torch.cuda.is_available():
        print('cuda     NOT available -> training would run on CPU. On the RTX 5050 laptop this means the wrong torch '
              'build (CPU-only) or a missing NVIDIA driver; see HANDOFF.md "Install".')
        sys.exit(0 if ok else 1)
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f'gpu      {name}, compute capability {cap[0]}.{cap[1]}, {free / 2**30:.1f} of {total / 2**30:.1f} GiB free')
    arch = f'sm_{cap[0]}{cap[1]}'
    archs = torch.cuda.get_arch_list()
    if arch not in archs:
        print(f'  !! this torch build has no kernels for {arch} (has: {" ".join(archs)}). RTX 50xx (Blackwell) needs a '
              'torch build for CUDA 12.8+, see HANDOFF.md "Install".'); ok = False
    try:
        x = torch.randn(1024, 1024, device='cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            y = (x @ x).float().sum().item()
        torch.cuda.synchronize()
        print(f'kernel   matmul + autocast ok ({y:.1f})')
    except Exception as e:  # e.g. "no kernel image is available for execution on the device"
        print(f'  !! a GPU kernel failed: {e}'); ok = False
    print('ENV OK' if ok else 'ENV HAS PROBLEMS (see !! lines)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
