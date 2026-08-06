#!/bin/bash
# =============================================================================
# dsv4-concurrency-admission — fix DSV4-0731 5-group KV admission accounting
# =============================================================================
# ROOT CAUSE (upstream vLLM issue #51041, open):
# DeepSeek-V4-Flash-0731 uses a 5-group hybrid KV cache (full MLA @ bs=256,
# C4 @ bs=64, C128 @ bs=64, two SWA/draft groups @ bs=4 and bs=8). The pool
# is sized in 256-token blocks (num_gpu_blocks = available_memory //
# bytes_per_256block), but the scheduler's admission path sums each group's
# RAW block count:
#
#   kv_cache_coordinator.get_num_blocks_to_allocate:
#       return sum(blocks_by_group)     # mixed block sizes!
#
# A 35K-token request then needs 137 (bs=256) + 545 + 545 (bs=64) +
# 8713 (bs=4) + 4357 (bs=8) = 14,297 blocks against a ~12,467-block pool
# -> only ~1 request fits -> "capacity" waiting at 2-3 concurrent -> the
# c>1 decode collapse. The tiny-block draft groups dominate the count even
# though their real memory is small.
#
# FIX: normalize every group's block count to the scheduler block size
# (256) before summing. A bs=4 block is 1/64 of a scheduler block, so
# 8713 raw blocks -> 8713 * 4/256 = 136 scheduler-equivalent blocks.
# Now a 35K request needs ~681 scheduler blocks -> ~18 fit. Matches the
# actual memory math (the pool sizer already weights blocks by bytes via
# _bucket_layers_by_page_size).
#
# Mod is idempotent and self-verifying (byte-compile + anchor check).
# =============================================================================
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
VLLM="$PYTHON_ROOT/vllm"
TARGET="$VLLM/v1/core/kv_cache_coordinator.py"

if [ ! -f "$TARGET" ]; then
  echo "[dsv4-concurrency-admission] missing $TARGET — wrong vLLM tree?" >&2
  exit 1
fi

python3 - "$VLLM" <<'PY'
import py_compile
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = root / "v1/core/kv_cache_coordinator.py"
text = target.read_text()

marker = "# dsv4-concurrency-admission"
if marker in text:
    print("[dsv4-concurrency-admission] skip (already applied)")
else:
    old = """        if self.lockstep_mla_allocations:
            return max(blocks_by_group, default=0)
        return sum(blocks_by_group)
"""
    new = """        if self.lockstep_mla_allocations:
            return max(blocks_by_group, default=0)
        # dsv4-concurrency-admission: normalize per-group block counts to the
        # scheduler block size (the pool is sized in scheduler blocks, but
        # DSV4-0731's 5-group hybrid cache reports raw counts at bs=4/8/64
        # which dominate the sum and cap admission at ~1 request). See
        # vllm issue #51041. Divide by (group_block_size / scheduler_bs).
        if self.scheduler_block_size > 0:
            scheduler_bs = self.scheduler_block_size
            blocks_by_group = [
                max(1, int(round(b * manager.block_size / scheduler_bs)))
                if manager.block_size != scheduler_bs
                else b
                for b, manager in zip(blocks_by_group, self.single_type_managers)
            ]
        return sum(blocks_by_group)
"""
    assert old in text, "anchor missing: coordinator sum path"
    target.write_text(text.replace(old, new, 1))
    print("[dsv4-concurrency-admission] patched coordinator sum normalization")

py_compile.compile(str(target), doraise=True)
print("[dsv4-concurrency-admission] byte-compile OK")
PY

echo "[dsv4-concurrency-admission] done"
