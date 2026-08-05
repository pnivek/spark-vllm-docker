# DSV4 Runtime Mods — eugr-compatible + VLLM_MODS bridge

This fork packages DSV4 runtime mods the **eugr way** (plain `run.sh` scripts in
`mods/`) with one additive bridge so Komodo deployments can toggle them via an
env var. eugr's native flow is fully preserved — no regression.

## The mod system (stock eugr)

eugr's mods are `mods/<name>/run.sh` scripts that patch installed site-packages
at container start. Selection is **not** env-driven:

- Recipe field: `mods:` list in `recipes/*.yaml`
- CLI: `run-recipe.py --apply-mod mods/<name>` → `launch-cluster.sh --apply-mod`

`launch-cluster.sh` copies each mod into the container and executes its
`run.sh` **before** launching vLLM. No Dockerfile involvement.

## What this fork adds (additive only)

1. **Mods baked into the image** — `COPY mods /opt/mods` in the Dockerfile, so
   the image is self-contained (no host-side files, no copy tricks).
2. **Toggle loader** — `mods/run_mods.sh` applies `$VLLM_MODS` (space-separated
   mod names) at container start. Empty/unset = fully stock.
3. **Everything else stays eugr-native** — mods remain plain `run.sh` scripts in
   `mods/`; `launch-cluster.sh --apply-mod mods/<name>` still works untouched.

## Toggle usage

| `VLLM_MODS` | Result |
|---|---|
| (unset / "") | stock eugr image (fp8 KV, concurrency ~1.5x) |
| `dsv4-kv-memory-estimate` | honest per-request KV estimate (~9x concurrency) |
| `nvfp4-dsv4-kv` | enable nvfp4_ds_mla KV (dtype gate + alignment + envelope) |
| `VLLM_NVFP4_ENVELOPE=432` (env, with above) | true NVFP4 record (432B) instead of fp8-compatible 584B page — ~1.35x more pool capacity; verify read-kernel path first |
| `dsv4-kv-memory-estimate nvfp4-dsv4-kv` | current production config |

Run order: memory-estimate first (independent), then nvfp4.

## Deployment paths

### Path A — Komodo Stack (compose, current production)

```yaml
environment:
  - VLLM_MODS=dsv4-kv-memory-estimate nvfp4-dsv4-kv
command:
  - |
    bash /opt/mods/run_mods.sh && \
    vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
      --kv-cache-dtype nvfp4_ds_mla ...
```

### Path B — Komodo Deployment (image + command, no compose)

A `Deployment` resource is a `docker run` wrapper: image + `command` (replaces
CMD) + `extra_args` (raw docker flags) + env + host network. Since mods are
baked into the image, the command is self-contained:

```yaml
image: 192.168.0.181:5000/pnivek/vllm-node-b12x:latest
network: host
restart: unless-stopped
command: >-
  bash /opt/mods/run_mods.sh &&
  vllm serve deepseek-ai/DeepSeek-V4-Flash-0731
  --host 0.0.0.0 --port 8000 --trust-remote-code
  --tensor-parallel-size 2 --kv-cache-dtype nvfp4_ds_mla
  --block-size 256 --max-model-len 1048576
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85
  --enable-prefix-caching --tokenizer-mode deepseek_v4
  --moe-backend b12x --linear-backend b12x --attention-backend B12X_MLA_SPARSE
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","attention_backend":"B12X_MLA_SPARSE"}'
  --distributed-executor-backend mp --nnodes 2 --node-rank 0
  --master-addr 192.168.0.172 --master-port 29506
environment: |
  VLLM_MODS=dsv4-kv-memory-estimate nvfp4-dsv4-kv
  HF_TOKEN=[[HF_TOKEN]]
  HF_HOME=/cache/huggingface
  HF_HUB_OFFLINE=1
  VLLM_HOST_IP=192.168.0.172
  NCCL_IB_HCA=<ib-if> NCCL_SOCKET_IFNAME=<eth-if>
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
extra_args:
  - --gpus=all
  - --ipc=host
  - --privileged
volumes: |
  /home/pnivek/.cache/huggingface:/cache/huggingface
```

A 2-node cluster = **two Deployment resources** (one per server) with matching
`--nnodes 2`, `--node-rank 0|1`, and the same `--master-addr`/`--master-port`.

## Reproducibility

Everything required lives in this repo:

- Mods: `mods/<name>/run.sh` (+ `mods/run_mods.sh` loader)
- Image: Dockerfile `COPY mods /opt/mods` (no external files needed)
- Deploy: either a compose Stack or an image Deployment, both referencing only
  the image + `VLLM_MODS`

A dev with Komodo + this repo resource can build the image and deploy with no
host-side staging, matching how eugr's own devs would run it.
