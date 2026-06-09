#!/bin/bash
# Quick status of an in-flight RSNA array job.
# Usage: bash scripts/check_rsna_progress.sh [arrayjobid]
# If no arg, picks the latest qoq_rsna_* output dir in outputs/.

set -uo pipefail

ARRAYJOB="${1:-$(ls -1dt outputs/qoq_rsna_*_shard0 2>/dev/null \
    | head -1 | sed -E 's|.*qoq_rsna_([0-9]+)_shard0|\1|')}"
if [[ -z "$ARRAYJOB" ]]; then
  echo "no array job id given and none found in outputs/" >&2
  exit 1
fi
echo "array job: $ARRAYJOB"
echo

# --- squeue (still running?) ---
echo "=== squeue ==="
squeue -u "$USER" 2>/dev/null | awk -v j="$ARRAYJOB" 'NR==1 || $1 ~ "^"j"_"'
echo

# --- per-shard tqdm progress (parsed from .err) ---
echo "=== per-shard progress ==="
printf "%-6s %-7s %-12s %-10s %-12s %-12s %-9s\n" "shard" "done?" "studies" "pct" "elapsed" "eta" "s/study"
printf "%-6s %-7s %-12s %-10s %-12s %-12s %-9s\n" "-----" "-----" "-------" "---" "-------" "---" "-------"

shopt -s nullglob
total_done=0; total_studies=0; rates=()
for d in outputs/qoq_rsna_${ARRAYJOB}_shard*/; do
  s=$(basename "$d" | sed -E 's|.*shard([0-9]+).*|\1|')
  err="logs/qoq_rsna_${ARRAYJOB}_${s}.err"

  csv_done=""
  [[ -f "${d}per_subject_scores.csv" ]] && csv_done="yes"

  # last tqdm-style "studies:" line — format e.g.
  # studies:  3%|...| 145/4349 [02:23:00<74:50:12, 64.10s/it]
  line=$(tac "$err" 2>/dev/null | grep -m1 'studies:' || true)
  if [[ -n "$line" ]]; then
    pct=$(  echo "$line" | grep -oE '[0-9]+%' | head -1)
    cur=$(  echo "$line" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f1)
    tot=$(  echo "$line" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f2)
    el=$(   echo "$line" | grep -oE '\[[0-9:]+<' | tr -d '[<')
    eta=$(  echo "$line" | grep -oE '<[0-9:]+,' | tr -d '<,')
    rate=$( echo "$line" | grep -oE '[0-9.]+s/it' | head -1 | sed 's|s/it||')
    [[ -n "$rate" ]] && rates+=("$rate")
    [[ -n "$cur"  ]] && total_done=$(( total_done + cur ))
    [[ -n "$tot"  ]] && total_studies=$(( total_studies + tot ))
  else
    pct=""; cur=""; tot=""; el=""; eta=""; rate=""
  fi

  printf "%-6s %-7s %-12s %-10s %-12s %-12s %-9s\n" \
    "$s" "${csv_done:-no}" "${cur:-?}/${tot:-?}" "${pct:-?}" \
    "${el:-?}" "${eta:-?}" "${rate:-?}"
done

# --- roll-up ---
echo
if (( total_studies > 0 )); then
  pct_all=$(( 100 * total_done / total_studies ))
  echo "overall: $total_done / $total_studies studies (${pct_all}%)"
fi
if (( ${#rates[@]} > 0 )); then
  avg=$(printf '%s\n' "${rates[@]}" | awk '{s+=$1} END{printf "%.1f", s/NR}')
  echo "avg s/study across shards: $avg"
fi

# --- error / traceback scan ---
echo
echo "=== errors (excluding tqdm + FutureWarning) ==="
for d in outputs/qoq_rsna_${ARRAYJOB}_shard*/; do
  s=$(basename "$d" | sed -E 's|.*shard([0-9]+).*|\1|')
  err="logs/qoq_rsna_${ARRAYJOB}_${s}.err"
  hits=$(grep -E '(Traceback|Error|RuntimeError|OOM|raise |slurmstepd:|CANCELLED|TIMEOUT)' "$err" 2>/dev/null | head -3)
  [[ -n "$hits" ]] && { echo "--- shard $s ---"; echo "$hits"; }
done
echo "(nothing above = all clean)"
