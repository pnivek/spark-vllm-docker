#!/bin/bash
# =============================================================================
# mods/run_mods.sh — toggle loader for DSV4 b12x runtime mods
#
# Runs every mod listed in $VLLM_MODS (space-separated) in order. Each mod
# is idempotent: re-running on an already-patched install is a no-op, and
# skipping it leaves the stock install untouched. Set VLLM_MODS="" (or unset)
# to run completely stock.
#
# Usage (in stack command, before `vllm serve ...`):
#   VLLM_MODS="dsv4-kv-memory-estimate nvfp4-dsv4-kv" bash /opt/mods/run_mods.sh
#
# Mods currently available (drop run.sh files into /opt/mods/<name>/):
#   dsv4-kv-memory-estimate  — fix per-request KV memory estimate (fp8+nvfp4)
#   nvfp4-dsv4-kv            — enable nvfp4_ds_mla KV (dtype gate+alignment)
# =============================================================================
set -euo pipefail

MODS_DIR="${MODS_DIR:-/opt/mods}"
ENABLED="${VLLM_MODS:-}"

if [ -z "$ENABLED" ]; then
  echo "[run_mods] VLLM_MODS empty/unset — running STOCK (no mods applied)"
  exit 0
fi

for mod in $ENABLED; do
  SCRIPT="$MODS_DIR/$mod/run.sh"
  if [ ! -x "$SCRIPT" ]; then
    echo "[run_mods] ERROR: mod '$mod' not found/executable at $SCRIPT" >&2
    exit 1
  fi
  echo "[run_mods] applying mod: $mod"
  bash "$SCRIPT"
done

echo "[run_mods] all enabled mods applied"
