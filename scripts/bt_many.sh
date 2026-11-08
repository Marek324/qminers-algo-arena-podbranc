#!/bin/bash
# usage: scripts/bt_many.sh <tape.csv> <outdir> <jobs> strategy1.py strategy2.py ...
# Backtests each strategy on the tape (--no-logs) in parallel and prints pnl / penalty / inventory quantiles.
tape=$1; out=$2; jobs=$3; shift 3
mkdir -p "$out"
printf '%s\n' "$@" | xargs -P "$jobs" -I{} sh -c '.venv/bin/exchange-simulator --strategy {} --backtest '"$tape"' --log-dir '"$out"'/$(basename {} .py) --no-logs > /dev/null 2>&1'
for s in "$@"; do
  n=$(basename "$s" .py)
  python3 -c "import json;r=json.load(open('$out/$n/result.json'));print(f\"{'$n':24s} pnl {r['pnl']:>11,.0f}  penalty {r['breach_penalty']:>9,.0f}  q95 {r['abs_inventory_q95']:>4} q997 {r['abs_inventory_q997']:>4}\")" 2>/dev/null || echo "$n FAILED"
done
