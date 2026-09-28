# Hardware — GPU memory budget

**Caveat up front:** exact parameter counts, layer counts, and attention
architecture (hidden size, number of KV heads, whether Multi-head Latent
Attention or plain GQA is used) for the specific model builds in
`src/bedrock/server/defaults/models.yaml` were not independently verifiable against a primary
model card at the time of this pass — see docs/server/ARCHITECTURE.md section 3.
The weight-size numbers below are computed from published parameter counts
(reliable to within precision-format rounding); the **KV-cache** numbers are
order-of-magnitude estimates using standard formulas and should be
recomputed from each model's actual `config.json` (hidden size, layer
count, KV heads, and any latent-attention compression factor) before
finalizing a hardware order.

## Method

For a dense transformer:

```
weight_bytes = total_params × bytes_per_param
kv_cache_bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_param
```

For an MoE model, **weight storage still requires every expert** (routing
picks a subset per token, but all experts must be resident in memory), so
`weight_bytes` uses total params, not active params. KV cache scales with
active-path attention dimensions only, so MoE doesn't inflate it further
compared to a dense model of similar hidden size.

`bytes_per_param`: FP8 = 1, INT4 = 0.5, BF16/FP16 = 2.

## Per-model budget (at src/bedrock/server/defaults/models.yaml defaults)

| Model | Role | Total params | Active params | Precision | Weight VRAM | Context (configured) | Est. KV cache @ full context* |
|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash-0731 | Director | ~284B | ~13B | FP8 | ~284 GB | 131,072 | Not independently computable without the published attention config — DeepSeek's architecture line uses compressed/latent attention specifically to keep this small at long context; **budget 20–40 GB as a placeholder and confirm against config.json.** |
| Qwen3.8-27B | Specialist | ~27B | ~27B (dense) | FP8 | ~27 GB | 65,536 | ~15–25 GB (standard GQA estimate; confirm num_kv_heads/head_dim) |
| Qwen3.6-35B-A3B | Runner | ~35B | ~3B | FP8 | ~35 GB | 32,768 | ~5–10 GB |

\* KV cache scales with **concurrent** sequences × context length actually used, not the configured max — these figures assume one long-context sequence in flight per model as a worst-case per-model ceiling, not a multi-tenant serving estimate. Real KV usage under concurrent chat sessions will be smaller per-session but additive across sessions — vLLM/SGLang's PagedAttention-style allocators handle this dynamically; the budget below reserves headroom for it.

**Running total (weights only): ~346 GB.** Add reserved KV-cache headroom
(say 100–150 GB pooled across all three engines to be safe pending the
config.json confirmation above) and vLLM/SGLang's own overhead (CUDA
graphs, activation buffers — typically 10–20 GB per engine instance): call
it **~500–550 GB of GPU memory to run all three models simultaneously at
FP8.**

## Does it fit a ~$300K server?

A representative ~$300K on-prem AI server in this class is an 8-GPU node
of 80 GB-class datacenter GPUs (e.g. H100 80GB SXM) — **640 GB of total GPU
memory** — or a smaller number of higher-memory GPUs (e.g. 4–8× H200
141GB, or newer 192GB-class parts) at a similar price point depending on
market pricing at time of purchase.

**Verdict: fits, with headroom, on an 8×80GB-class node at FP8** — ~550 GB
needed against 640 GB available. It does **not** have comfortable headroom
for the Director's Pro sibling (`DeepSeek-V4-Pro-0813`, ~1.6T total params
→ ~1.6 TB at FP8, ~800 GB even at INT4) on a single node; that model is
listed in `src/bedrock/server/defaults/models.yaml` as a fallback for a second node, not the
default.

Fallbacks, in order of preference, if actual KV-cache/overhead numbers come
in higher than the placeholder above or if the server spec comes in below
640 GB of GPU memory:

1. **Quantize the Director further** (INT4 for DeepSeek-V4-Flash) — cuts
   its weight footprint from ~284 GB to ~142 GB, buying ~140 GB of extra
   headroom at some quality cost on the hardest tasks. Since the Director
   is called rarely, this is the first lever to pull.
2. **Reduce configured context length** for the Director below 131,072 in
   `src/bedrock/server/defaults/models.yaml` — directly shrinks whatever the real KV-cache
   number turns out to be; the Director doesn't need 1M-token context for
   most planning/review calls.
3. **Share GPUs across engines with time-slicing** (MPS or MIG on the same
   physical GPUs) if the three models' peak-usage windows don't overlap
   much — the Runner and Specialist run far more often than the Director,
   so the Director's engine can be sized smaller and lean on request
   queuing during its rare, expensive calls.
4. **Swap the Runner down** to `Qwen/Qwen3.5-0.8B` (already listed as an
   alternative in `src/bedrock/server/defaults/models.yaml`) — frees tens of GB with negligible
   quality impact on classification/formatting-only work.
5. **Two-node split** as a last resort — Director on its own smaller node,
   Specialist+Runner on the primary node — if the company's budget allows
   a second, cheaper GPU box rather than fitting everything on one.

## Action item before procurement

Pull the actual `config.json` for each final model choice (hidden size,
layer count, KV head count, and — critically — confirm whether
DeepSeek-V4-Flash's attention is the latent/compressed style referenced in
its announcement) and recompute the KV-cache column above precisely. The
weight-VRAM numbers are solid; the KV-cache numbers are the ones to
re-verify before signing a purchase order.
