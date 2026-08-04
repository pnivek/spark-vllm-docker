#!/bin/bash
# =============================================================================
# mods/dsv4-entrypoint.sh — image entrypoint: apply $VLLM_MODS then exec "$@"
#
# Replaces the NVIDIA default entrypoint for DSV4 images so a Deployment
# command is just `vllm serve ...` (no && chaining needed — Komodo splits
# command strings on &&). The loader is a no-op when VLLM_MODS is unset,
# preserving stock eugr behavior.
# =============================================================================
set -euo pipefail

if [ -n "${VLLM_MODS:-}" ]; then
  bash /opt/mods/run_mods.sh
fi

exec "$@"
