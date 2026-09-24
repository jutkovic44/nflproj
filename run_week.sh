#!/usr/bin/env bash
# Weekly routine: refresh data, project the coming week, save a dated CSV.
set -e
SEASON=${1:?usage: ./run_week.sh SEASON WEEK}
WEEK=${2:?usage: ./run_week.sh SEASON WEEK}
python3 -m nflproj.cli refresh
python3 -m nflproj.cli project --season "$SEASON" --week "$WEEK" \
  --output "projections_${SEASON}_wk${WEEK}.csv"
