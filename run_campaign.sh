#!/usr/bin/env bash
#
# Run both circuit families, one after the other, for this machine's slice.
#
#   ./run_campaign.sh 1/2          # machine A of two
#   ./run_campaign.sh 2/2          # machine B
#   ./run_campaign.sh 1/1 -w 8     # one machine, custom worker count
#
# Sequential rather than concurrent on purpose: both families are CPU-bound, so
# running them together only splits the same cores and makes each slower. It also
# keeps the amplifier -- the expensive half -- from being starved.
#
# Each slice writes its own CSV. Concatenate them when every machine has
# finished; the rows carry family, variant, target and seed, so order does not
# matter.
set -euo pipefail

cd "$(dirname "$0")"
SHARD=${1:?uso: ./run_campaign.sh <i/N> [argumentos extras para campaign.py]}
shift || true

SEEDS=${SEEDS:-100}
TAG=$(echo "$SHARD" | tr '/' 'de')
RESULTS=results
mkdir -p "$RESULTS"

[ -x venv/bin/python ] || { echo "venv ausente — rode ./setup_env.sh primeiro" >&2; exit 1; }

# Default to physical cores plus a quarter; SMT and E-cores add throughput but
# not proportionally, and leaving a little headroom keeps the machine usable.
if ! printf '%s\n' "$@" | grep -qE '^-w$|^--workers'; then
  PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
  WORKERS=$(( PHYS + PHYS / 4 ))
  [ "$WORKERS" -gt "$(nproc)" ] && WORKERS=$(nproc)
  set -- -w "$WORKERS" "$@"
fi

started=$(date +%s)
echo "campanha iniciada $(date '+%F %T') | shard $SHARD | $SEEDS sementes | $*"

for family in filter amp; do
  log="$RESULTS/${family}_${TAG}.log"
  csv="$RESULTS/${family}_${TAG}.csv"
  echo
  echo "== $family =="
  echo "   log: $log"
  t0=$(date +%s)
  PYTHONPATH=. venv/bin/python -u campaign.py \
      --family "$family" --seeds "$SEEDS" --shard "$SHARD" \
      --data-csv "$csv" \
      --out-dir "$RESULTS/runs" "$@" 2>&1 | tee "$log"
  echo "   $family levou $(( ($(date +%s) - t0) / 60 )) min"
done

echo
echo "campanha concluida em $(( ($(date +%s) - started) / 60 )) min"
echo "resultados em $RESULTS/:"
ls -la "$RESULTS"/*.csv 2>/dev/null || true
echo
echo "quando todas as maquinas terminarem, junte os CSVs de mesma familia:"
echo "  head -1 filter_1de2.csv > filter_all.csv"
echo "  tail -q -n +2 filter_*de*.csv >> filter_all.csv"
