#!/usr/bin/env python
"""Task 15: Verify end-to-end experiment (smoke run)."""
import subprocess
import sys

print('=== Task 15: End-to-End Experiment Smoke Run ===')
print('Running smoke experiment via experiment_runner...')

# Run the smoke experiment
result = subprocess.run(
    [sys.executable, '-m', 'src.experiment_runner', '--smoke'],
    cwd='c:\\Users\\amank\\OneDrive\\Desktop\\streaming_data_proj-main',
    capture_output=False,
    timeout=600
)

if result.returncode == 0:
    print('\n✓ End-to-end experiment smoke run PASSED')
else:
    print(f'\n✗ End-to-end experiment smoke run FAILED with exit code {result.returncode}')
    sys.exit(1)
