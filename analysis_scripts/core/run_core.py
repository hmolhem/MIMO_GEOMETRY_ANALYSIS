"""
Dispatcher wrapper for canonical core analysis scripts.
Usage:
    python -m analysis_scripts.core.run_core run_benchmarks [args...]
This will execute the original script at core/analysis_scripts/run_benchmarks.py with forwarded arguments.
"""
import os
import sys
import runpy

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CORE_SCRIPTS_DIR = os.path.join(REPO_ROOT, 'core', 'analysis_scripts')

def find_script(name):
    # accept with or without .py
    candidate = name if name.endswith('.py') else name + '.py'
    path = os.path.join(CORE_SCRIPTS_DIR, candidate)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Core script not found: {candidate}")

def main():
    if len(sys.argv) < 2:
        print("Usage: run_core.py <script_name> [args...]")
        print("Example: python run_core.py run_benchmarks --help")
        sys.exit(1)
    script = sys.argv[1]
    script_path = find_script(script)
    # Forward argv (drop the dispatcher argument)
    sys.argv = [script_path] + sys.argv[2:]
    runpy.run_path(script_path, run_name='__main__')

if __name__ == '__main__':
    main()
