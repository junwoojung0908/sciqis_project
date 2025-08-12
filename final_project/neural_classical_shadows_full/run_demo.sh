#!/usr/bin/env bash
set -e
export PYTHONPATH=.
python scripts/demo_end_to_end.py
python scripts/prediction_panels.py
python scripts/run_shots_sweep.py --shots 50,100,200
python scripts/feature_sensitivity.py --topk 20
echo 'All done. Check outputs/.'
