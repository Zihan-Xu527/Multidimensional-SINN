#!/bin/bash
# Fetch the Section 4 Duffing saved inputs from the cluster into this directory.
#
# COPY ONLY. Nothing here trains, re-evaluates, or recomputes a statistic. No
# job is submitted; no remote file is written or deleted.
#
# RELATIVE LAYOUT IS PRESERVED, and it is the layout the entry points in
# README.md already expect:
#
#     duffing/runs/grid/            --rundir runs/grid
#     duffing/reports_val/          --repdir reports_val
#     duffing/reports_val_disp/     --repdir reports_val_disp   (contour figure)
#     duffing/din_eval_cells/       --cachedir din_eval_cells
#     duffing/data/                 --datadir data
#     duffing/din_eval.json         --json din_eval.json
#     duffing/timing_exclusive.json
#
# The paper's saved JSON/NPZ histories are intentional figure inputs.
# Large checkpoints remain excluded by the root .gitignore's *.pt/*.pth.
#
# Run this FROM this directory:
#
#     bash fetch_saved_inputs.sh            # items 1-5
#     bash fetch_saved_inputs.sh 1 2 4      # only those numbered items
#     bash fetch_saved_inputs.sh 7          # contour inputs (large; opt-in)
#     DRY=1 bash fetch_saved_inputs.sh      # list what would transfer
#
# The host and remote directory below are taken from the cluster scripts
# themselves (`reports_val.sh`, `gen_joint.sh`, `timing.sh` all `cd` to REMOTE);
# the login host was supplied by the author. Override either if yours differ:
#
#     HOST=zxu64@login.rc.ucmerced.edu REMOTE=/path/to/Duffing_dimer bash fetch_saved_inputs.sh

set -euo pipefail

HOST="${HOST:-zxu64@login.rc.ucmerced.edu}"
REMOTE="${REMOTE:-/home/zxu64/GIT/Multidimensional-SINN/MDSINN/Duffing_dimer}"
# --progress, not --info=progress2: macOS ships rsync 2.6.9, which has no
# --info flag at all. --progress works on both old and new rsync.
RS=(rsync -av --prune-empty-dirs --progress)
[ -n "${DRY:-}" ] && RS+=(--dry-run)

ITEMS=("$@")

# REJECT ANYTHING THAT IS NOT AN ITEM NUMBER. Without this the script
# silently transfers nothing: zsh does not treat `#` as a comment in an
# interactive shell, so a pasted trailing comment arrives here as
# arguments and every want() test fails.
for a in "${ITEMS[@]:-}"; do
  case "$a" in
    1|2|3|4|5|6|7) ;;
    "") ;;
    *) echo "fetch_saved_inputs.sh: not an item number: '$a'" >&2
       echo "usage: bash fetch_saved_inputs.sh [1..7]   (no arguments = items 1-5)" >&2
       echo "do not paste a trailing # comment: zsh passes it as an argument." >&2
       exit 2 ;;
  esac
done

# No arguments means items 1-5 (the six requested items minus the
# un-automatable item 6). Item 7 is the large contour-renderer input and
# is never fetched unless it is named explicitly.
want() {
  if [ "${#ITEMS[@]}" -eq 0 ]; then
    [ "$1" = "7" ] && return 1 || return 0
  fi
  local a
  for a in "${ITEMS[@]}"; do [ "$a" = "$1" ] && return 0; done
  return 1
}

# --- 1. L2H10 balancing histories and retained-run records (140 runs) --------
# 20 runs at lambda=4, 10 at each of 0, 0.5, 1, 2, 8; both balancing settings.
# Expect 140 .npz + 140 .json = 280 files.
if want 1; then
  echo "### item 1: runs/grid"
  mkdir -p runs/grid
  "${RS[@]}" \
    --include='hist_nf3_lbon_din2_L2H10_lam*_s*.npz' \
    --include='hist_nf3_lboff_din2_L2H10_lam*_s*.npz' \
    --include='run_nf3_lbon_din2_L2H10_lam*_s*.json' \
    --include='run_nf3_lboff_din2_L2H10_lam*_s*.json' \
    --exclude='*' \
    "$HOST:$REMOTE/runs/grid/" runs/grid/
fi

# --- 2. validation-anchored reports (24) ------------------------------------
# reports_val/, NOT reports/. The two anchors must not be mixed.
if want 2; then
  echo "### item 2: reports_val"
  mkdir -p reports_val
  "${RS[@]}" \
    --include='report_nf0_lbon_din2_L2H10_lam*.json' \
    --include='report_nf1_lbon_din2_L2H10_lam*.json' \
    --include='report_nf2_lbon_din2_L2H10_lam*.json' \
    --include='report_nf3_lbon_din2_L2H10_lam*.json' \
    --exclude='*' \
    "$HOST:$REMOTE/reports_val/" reports_val/
fi

# --- 3. input-dimension evaluation (1 + 18) ---------------------------------
if want 3; then
  echo "### item 3: din_eval.json + din_eval_cells"
  "${RS[@]}" "$HOST:$REMOTE/din_eval.json" .
  mkdir -p din_eval_cells
  "${RS[@]}" --include='*.json' --exclude='*' \
    "$HOST:$REMOTE/din_eval_cells/" din_eval_cells/
fi

# --- 4. reference archives and target statistics (6 + 6 + 6) ----------------
# These fix standardization, projection scales, density grids and bandwidths.
# Every saved model must be loaded against them.
if want 4; then
  echo "### item 4: data/reference_lam*, data/target_stats_lam*"
  mkdir -p data
  "${RS[@]}" \
    --include='reference_lam*.npz' \
    --include='reference_lam*.json' \
    --include='target_stats_lam*.npz' \
    --exclude='*' \
    "$HOST:$REMOTE/data/" data/
fi

# --- 5. exclusive-GPU timing result -----------------------------------------
if want 5; then
  echo "### item 5: timing_exclusive.json"
  "${RS[@]}" "$HOST:$REMOTE/timing_exclusive.json" .
fi

# --- 6. S1 late-training-loss/success table --------------------------------
# The required inputs are already the item-1 histories and run records.
# table_balancing.py exports this table locally, without loading a model.
# No remote search or transfer is needed for item 6.
if want 6; then
  echo "### item 6: already covered by item 1; run python table_balancing.py locally"
fi

# --- contour renderer inputs -------------------------------------------------
# traj_lam4.npy is the full raw archive for lambda = 4 and is large; the four
# display ensembles and their four report JSONs come from reports_val_disp/
# (gen_joint.sh), not reports_val/. fig_joint.py reads both from --repdir.
# Fetch these only if the contour figure is being rebuilt.
if want 7; then
  echo "### item 7: contour renderer inputs (large)"
  mkdir -p data reports_val_disp
  "${RS[@]}" "$HOST:$REMOTE/data/traj_lam4.npy" data/
  "${RS[@]}" \
    --include='report_nf[0-3]_lbon_din2_L2H10_lam4.json' \
    --include='gen_nf[0-3]_lbon_din2_L2H10_lam4_s*.npz' --exclude='*' \
    "$HOST:$REMOTE/reports_val_disp/" reports_val_disp/
fi

echo
echo "=== transferred, by item ==="
for d in runs/grid reports_val reports_val_disp din_eval_cells data; do
  # Assigned first rather than inlined in the printf. The inline form was
  # once suspected of tripping macOS bash 3.2; that was tested and is NOT
  # true (bash 3.2.57 parses it fine). This form is kept only because it
  # is easier to read, not as a workaround for anything.
  if [ -d "$d" ]; then
    n=$(find "$d" -type f | wc -l)
    printf '%-20s %d files\n' "$d" "$n"
  fi
done
for f in din_eval.json timing_exclusive.json; do
  [ -f "$f" ] && printf '%-20s present\n' "$f"
done
echo
echo "Expected: runs/grid 280, reports_val 24, din_eval_cells 18, data 18"
echo "(+ reports_val_disp 8 files: four JSON reports and four display ensembles,"
echo " plus data/traj_lam4.npy if item 7 was fetched)"
