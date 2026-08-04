#!/bin/bash
# =============================================================================
# nvfp4-dsv4-kv — enable nvfp4_ds_mla KV cache on DeepSeek-V4 b12x
# =============================================================================
# DEPENDS ON: dsv4-kv-memory-estimate (the concurrency fix) for the pool to
# size correctly, but applies independently of it.
#
# The base image already ships the nvfp4 infrastructure (CacheDType entry,
# concat_and_cache_nvfp4_mla writer + .so, B12X canonicalizer). What the
# stock code blocks:
#   1. dtype gate: _resolve_dsv4_kv_cache_dtype hard-asserts fp8 only.
#   2. page alignment: packed uint8 layouts need 576B (b12x kernels hardcode
#      it); stock gives nvfp4 512B -> 192B/block storage overrun crash.
#   3. backend dtype/shape: DeepseekV4FlashMLABackend doesn't advertise
#      nvfp4_ds_mla nor shape the 584B record.
#   4. page size: deepseek_v4 nvfp4 spec must be storage_block_size*584 to
#      match what the b12x compressed-MLA kernels read.
#
# After this mod, run vllm with: --kv-cache-dtype nvfp4_ds_mla
# =============================================================================
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
VLLM="$PYTHON_ROOT/vllm"

NEEDED_FILES=(
  "$VLLM/models/deepseek_v4/attention.py"
  "$VLLM/models/deepseek_v4/sparse_mla.py"
  "$VLLM/v1/attention/backends/mla/sparse_swa.py"
  "$VLLM/v1/kv_cache_interface.py"
)
for f in "${NEEDED_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "[nvfp4-dsv4-kv] missing $f — wrong vLLM tree?" >&2
    exit 1
  fi
done

python3 - "$VLLM" <<'PY'
import py_compile
import sys
from pathlib import Path

root = Path(sys.argv[1])


def replace(path: str, old: str, new: str, what: str) -> None:
    p = root / path
    text = p.read_text()
    if new in text:
        print(f"[nvfp4-dsv4-kv] skip  {what} (already applied)")
        return
    if old not in text:
        raise SystemExit(f"[nvfp4-dsv4-kv] FAIL {what}: anchor missing in {path}")
    p.write_text(text.replace(old, new, 1))
    print(f"[nvfp4-dsv4-kv] ok    {what} -> {path}")


# ---- 1. dtype gate: accept nvfp4/nvfp4_ds_mla in the packed-uint8 path ----
replace(
    "models/deepseek_v4/attention.py",
    """    if use_fp8_ds_mla_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, "
            f"got {kv_cache_dtype}"
        )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8
""",
    """    if use_fp8_ds_mla_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        # nvfp4_ds_mla: packed NVFP4 MLA record (B12X reads natively).
        if kv_cache_dtype in ("nvfp4", "nvfp4_ds_mla"):
            if cache_config is not None:
                cache_config.cache_dtype = "nvfp4_ds_mla"
            kv_cache_dtype = "nvfp4_ds_mla"
            logger.info_once("Using DeepSeek V4 nvfp4_ds_mla KV cache format.")
            return kv_cache_dtype, torch.uint8
        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, "
            f"got {kv_cache_dtype}"
        )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8
""",
    "dtype gate: nvfp4 accepted",
)

# ---- 2. alignment: 576B for nvfp4 in the main MLA spec ----
replace(
    "models/deepseek_v4/attention.py",
    """        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
""",
    """        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        uses_packed_uint8_layout = uses_fp8_ds_mla_layout or self.kv_cache_dtype in (
            "nvfp4",
            "nvfp4_ds_mla",
        )
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_packed_uint8_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_packed_uint8_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
""",
    "alignment: 576B for nvfp4 (main MLA)",
)

# ---- 3. alignment: 576B for nvfp4 in the SWA cache spec ----
replace(
    "v1/attention/backends/mla/sparse_swa.py",
    """        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.cache_config.cache_dtype),
        )
""",
    """        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
        uses_packed_uint8_layout = uses_fp8_ds_mla_layout or self.cache_config.cache_dtype in (
            "nvfp4",
            "nvfp4_ds_mla",
        )
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_packed_uint8_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.cache_config.cache_dtype),
        )
""",
    "alignment: 576B for nvfp4 (SWA cache)",
)

# ---- 4. backend: advertise nvfp4_ds_mla + KV shape branch ----
replace(
    "models/deepseek_v4/sparse_mla.py",
    """    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]
""",
    """    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
        "nvfp4_ds_mla",
    ]
""",
    "backend advertises nvfp4_ds_mla",
)

replace(
    "models/deepseek_v4/sparse_mla.py",
    """        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        else:
            return (num_blocks, block_size, head_size)
""",
    """        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        if cache_dtype_str == "nvfp4_ds_mla":
            # Keep DeepSeek V4's proven 584-byte cache envelope so hybrid
            # MLA/SWA grouping can proceed while testing nvfp4_ds_mla.
            return (num_blocks, block_size, 584)
        else:
            return (num_blocks, block_size, head_size)
""",
    "backend KV shape: nvfp4 envelope",
)

# ---- 5. page size: 584B envelope for deepseek_v4 nvfp4 ----
replace(
    "v1/kv_cache_interface.py",
    """            if self.model_version == "deepseek_v4":
                return self.storage_block_size * 432
            if self.model_version == "glm_fp8_rope":
""",
    """            if self.model_version == "deepseek_v4":
                # Match the fp8_ds_mla 584B envelope the b12x compressed-MLA
                # kernels read; true 432B NVFP4 record is follow-up work.
                return self.storage_block_size * 584
            if self.model_version == "glm_fp8_rope":
""",
    "page size: 584B envelope for deepseek_v4 nvfp4",
)

print("[nvfp4-dsv4-kv] verifying byte-compile...")
for rel in (
    "models/deepseek_v4/attention.py",
    "models/deepseek_v4/sparse_mla.py",
    "v1/attention/backends/mla/sparse_swa.py",
    "v1/kv_cache_interface.py",
):
    py_compile.compile(str(root / rel), doraise=True)
print("[nvfp4-dsv4-kv] all patches applied + byte-compiled OK")
PY

echo "[nvfp4-dsv4-kv] done. run vllm with: --kv-cache-dtype nvfp4_ds_mla"
