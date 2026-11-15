"""Command line: python -m urubamba {download,fit,evaluate,figures,all}."""
import argparse

from threadpoolctl import threadpool_limits

from .config import N_JOBS


def main():
    parser = argparse.ArgumentParser(prog="urubamba", description=__doc__)
    parser.add_argument("step", choices=["download", "fit", "evaluate", "figures", "all"])
    step = parser.parse_args().step
    with threadpool_limits(limits=N_JOBS):
        if step in ("download", "all"):
            from .data import download_all
            download_all()
        if step in ("fit", "all"):
            from .models import fit
            fit()
        if step in ("evaluate", "all"):
            from .evaluate import evaluate
            evaluate()
        if step in ("figures", "all"):
            from .figures import main as figures
            figures()


if __name__ == "__main__":
    main()
