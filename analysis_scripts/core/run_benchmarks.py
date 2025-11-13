#!/usr/bin/env python3
"""Canonical wrapper for run_benchmarks (core)."""
import os
import sys
import runpy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SCRIPT = os.path.join(REPO_ROOT, 'core', 'analysis_scripts', 'run_benchmarks.py')
if __name__ == '__main__':
    sys.argv = [SCRIPT] + sys.argv[1:]
    runpy.run_path(SCRIPT, run_name='__main__')
