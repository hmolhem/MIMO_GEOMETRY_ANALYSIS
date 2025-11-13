"""Thin CLI wrapper that delegates to the package-level runner.

This file is the backward-compatible entrypoint kept in place for existing
invocation paths. It simply imports and calls the refactored function in
`mimo_geom_analysis.runners`.
"""

from mimo_geom_analysis.runners import run_benchmarks


def main():
    # When invoked as a script we want to consume sys.argv; run_benchmarks will
    # default to parsing sys.argv when argv is None.
    run_benchmarks()


if __name__ == "__main__":
    main()


