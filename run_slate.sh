#!/usr/bin/env bash
# Run the next slate that has not kicked off: refresh data, project, rebuild
# the dashboard. Safe to run repeatedly — each run overwrites dashboard.html.
set -euo pipefail
cd "$(dirname "$0")"

LOG="runs/$(date +%Y%m%d_%H%M).log"
mkdir -p runs
{
  echo "=== $(date) ==="
  python3 -m nflproj.cli refresh
  # leakage audit gates the run: a failure stops the pipeline here
  python3 -m nflproj.cli audit
  # grade every completed game first, so the dashboard shows a current score
  python3 -m nflproj.cli grade --output accuracy.json
  python3 -m nflproj.cli project --next --dashboard dashboard.html
} 2>&1 | tee "$LOG"

# keep a dated copy of each slate's dashboard for later comparison
cp dashboard.html "runs/dashboard_$(date +%Y%m%d_%H%M).html"
# archive the slate's projections so future grading has a frozen pre-game record
cp projections_*.csv runs/ 2>/dev/null || true
