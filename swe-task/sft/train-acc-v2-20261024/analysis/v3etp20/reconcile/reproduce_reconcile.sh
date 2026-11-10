#!/bin/bash
# train-acc v3etp20 reconcile (2026-11-04): re-checks of the two verifiers' problems against the raw data.
# Read-only on probes / results / routers / DeepSWE runs and on every other analysis dir; writes only under reconcile/out.
# Python: /usr/bin/python3 -I -B, stdlib only.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
R=/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/sft/train-acc-v2-20261024/analysis/v3etp20/reconcile
PY="/usr/bin/python3 -I -B"
cd $R
$PY r_router.py                                             # tail workers: 660 served / 664 routed; 129 calls after 05:50
$PY r_export_scan.py sc14 etp20 v3etp5 v3etp10 v3etp20 --procs 12   # per-slot export scan -> out/export-<m>.jsonl.gz
$PY r_analyze.py                                            # '## Goal' 2/44, per-task groups, unreasoned, markup, loop units, non-Latin runs
$PY r_http_targeted.py                                      # deadline-kill calls (median 816), v3etp10 no-export slots, compaction responses
$PY r_unreasoned.py                                         # unreasoned-action tests on the export and all-slot bases
$PY r_deepswe.py                                            # DeepSWE from the raw runs: -6.02 pp vs base, kills 31/48 vs 19/37, endings
$PY r_loops.py                                              # loop outcome split under two definitions (lens records)
$PY r_reasoning_chars.py                                    # character-level reasoning length + KS to base
$PY r_uncommitted.py                                        # final texts of the 16 uncommitted endings
$PY r_empty_broken.py                                       # empty patches after a broken summary (28/36)
$PY r_runons.py                                             # argument run-ons, 'invalid' tool names, largest argument
$PY r_within_task.py                                        # tool-call-summary vs fine-summary rollouts within the same task
