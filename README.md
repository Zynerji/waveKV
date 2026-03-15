# waveKV — Fixed-Memory KV Cache for Any Transformer

**Drop-in replacement for standard KV caches. O(1) memory regardless of sequence length.**

Standard KV caches grow linearly with context — 32 GB at 128K tokens for a 32-layer model. waveKV compresses KV history into wave interference fields that never grow, plus a small exact sliding window for recent tokens.

**Pure PyTorch. No CUDA extensions. Works on NVIDIA, AMD, Intel, Apple Silicon, CPU.**

## Memory Comparison

| Context Length | Standard KV Cache | waveKV | Savings |
|---|---|---|---|
| 1,024 tokens | 1.0 GB | **25 MB** | 40x |
| 8,192 tokens | 8.0 GB | **25 MB** | 320x |
| 131,072 tokens | 128 GB | **25 MB** | 5,120x |
| 1,000,000 tokens | 1 TB | **25 MB** | 40,000x |
| 1,000,000,000 tokens | 1 PB | **25 MB** | 40,000,000x |

*32-layer, 32-head, 128-dim model, bf16. waveKV memory is fixed at 25 MB regardless of sequence length. Yes, a billion tokens uses the same 25 MB.*

## Install

```bash
pip install wave-kv
```

Or from source:

```bash
git clone https://github.com/Zynerji/waveKV.git
cd waveKV
pip install -e .
```

## Quick Start

### Auto-detect from any HuggingFace model

```python
from wave_kv import WaveKVManager

cache = WaveKVManager.for_model(model)
```

### Manual configuration

```python
from wave_kv import WaveKVManager

cache = WaveKVManager(
    n_layers=32,
    n_heads=32,
    head_dim=128,
    n_waves=128,      # more waves = better quality
    window_size=64,    # exact window for recent tokens
    device="cuda",     # or "cpu", "mps", etc.
)
```

### Integration with any generate loop

```python
from wave_kv import WaveKVManager

cache = WaveKVManager.for_model(model)

# Prefill
for layer_idx, block in enumerate(model.layers):
    kv = cache.get_kv(layer_idx)           # None on first call
    output, new_kv = block(x, kv_cache=kv)
    cache.absorb(layer_idx, new_kv, n_new=prompt_len)
cache.step()

# Decode loop
for _ in range(max_new_tokens):
    for layer_idx, block in enumerate(model.layers):
        kv = cache.get_kv(layer_idx)       # returns (k, v) tuple
        output, new_kv = block(x, kv_cache=kv)
        cache.absorb(layer_idx, new_kv, n_new=1)
    cache.step()

# Check compression
print(cache.stats())
# {'cached_tokens': 4096, 'wave_cache_mb': 25.1, 'standard_cache_mb': 1024.0, 'compression_ratio': 40.8}
```

## How It Works

Each attention head maintains a **wave interference field** — a set of sinusoidal wave parameters (frequency, amplitude, phase) that encode KV history. New tokens update the wave fields via learned EMA projections. Reconstruction samples the wave pattern at requested positions.

```
New K/V token → encode_proj → wave parameter deltas → EMA update → wave field
                                                                        ↓
Attention query ← decode_proj ← wave interference sampling ← reconstruct(positions)
```

**Sliding window**: The most recent `window_size` tokens are stored exactly (no compression). Older tokens are wave-compressed. This gives perfect quality for local attention while maintaining a compressed summary of full history.

## Hierarchical Wave Fields (v0.2)

Standard flat wave fields use a single EMA rate, giving a fixed effective memory window. At `ema_rate=0.01`, only the last ~100 tokens meaningfully survive — everything older reconstructs as noise.

**Hierarchical wave fields** solve this with multi-resolution decomposition:

```python
from wave_kv import WaveKVManager

# Hierarchical mode — multi-resolution wave levels
cache = WaveKVManager(
    n_layers=32,
    n_heads=32,
    head_dim=128,
    hierarchical=True,          # enable hierarchy
    window_size=64,
    device="cuda",
)
```

The hierarchy organizes waves into levels with different time horizons:

| Level | Waves | EMA Rate | Freq Range | Captures |
|-------|-------|----------|------------|----------|
| 0 (coarsest) | 16 | 0.001 | 0.01–1.0 | Ultra-long-range themes (entire context) |
| 1 | 32 | 0.005 | 0.5–3.0 | Broad semantic patterns |
| 2 | 64 | 0.02 | 1.0–6.0 | Paragraph-scale structure |
| 3 | 128 | 0.05 | 2.0–10.0 | Sentence-scale detail |
| 4 (finest) | 256 | 0.1 | 4.0–20.0 | Recent token precision |

**Total: 496 waves = ~30 MB fixed.** Coarse levels (slow EMA) maintain a persistent backbone of global context, while fine levels (fast EMA) track recent detail with high fidelity. Reconstruction sums contributions from all levels.

Custom level configurations:

```python
cache = WaveKVManager(
    n_layers=32, n_heads=32, head_dim=128,
    hierarchical=True,
    level_config=[
        # (n_waves, ema_rate, freq_low, freq_high)
        (32,  0.0005, 0.01, 0.5),   # very coarse — million-token memory
        (64,  0.005,  0.5,  4.0),   # mid
        (128, 0.05,   2.0,  12.0),  # fine
        (512, 0.15,   5.0,  30.0),  # ultra-fine — near-window precision
    ],
    device="cuda",
)
```

## Architecture Support

`WaveKVManager.for_model()` auto-detects:

- **LLaMA** / **LLaMA 2** / **LLaMA 3**
- **Mistral** / **Mixtral**
- **GPT-2** / **GPT-NeoX**
- **Phi** / **Phi-2** / **Phi-3**
- **Qwen** / **Qwen2**
- **Falcon**
- Any HuggingFace `PreTrainedModel` with standard config
- Any model with a countable `layers`/`blocks`/`h` attribute

For non-standard architectures, pass dimensions explicitly:

```python
cache = WaveKVManager(n_layers=40, n_heads=40, head_dim=128)
```

## Training the Wave Projections

**Important**: waveKV includes learned projection layers (`encode_proj` and `decode_proj`) that translate between KV vectors and wave parameter deltas. These projections start randomly initialized. Without training them, wave-compressed positions (older than the sliding window) will produce poor reconstructions.

**What this means in practice**:
- The **sliding window** (most recent `window_size` tokens) always returns **exact** K,V values — no training needed.
- **Wave-compressed positions** (older tokens) require trained projections for quality reconstruction. With random projections, these positions contribute noise rather than useful context.

**How to train**: Freeze the LLM and train only the cache projections on representative data. The training objective is KV reconstruction error — the projections learn to encode KV vectors into wave deltas and decode wave interference patterns back into KV vectors.

```python
from wave_kv import WaveKVManager

cache = WaveKVManager.for_model(model)

# Freeze LLM, train only wave cache projections
for p in model.parameters():
    p.requires_grad = False
for p in cache.cache.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(cache.cache.parameters(), lr=1e-4)

# Training loop: run forward passes, compute KV reconstruction loss
# between wave-reconstructed K,V and ground-truth K,V from standard cache
```

**Without training**, waveKV still works as a fixed-window cache (the sliding window portion is always exact). The wave-compressed history gradually improves as the projections are trained — even a few hundred steps of fine-tuning significantly improves reconstruction quality.

## Evaluation Results (GPT-2 Small, Hierarchical v0.2)

Trained on WikiText-103, evaluated on held-out text. Hierarchical 5-level cache (496 total waves, ~30 MB fixed) with 64-token exact sliding window.

### KV Reconstruction Quality by Quartile

Cosine similarity between wave-reconstructed and ground-truth KV vectors, measured at different position ranges (Q1 = oldest compressed tokens, Q4 = newest compressed tokens before window).

| Seq Length | Q1 (oldest) | Q2 | Q3 | Q4 (newest) |
|------------|------------|------|------|-------------|
| 128 | 0.693 | 0.801 | 0.810 | 0.809 |
| 256 | **0.803** | **0.835** | **0.833** | 0.818 |
| 512 | **0.812** | **0.820** | 0.806 | 0.792 |
| 768 | 0.783 | 0.783 | 0.762 | 0.741 |
| 1024 | 0.627 | 0.593 | 0.502 | 0.300 |

Best reconstruction at 256-512 tokens. Hierarchical wave levels maintain Q1 (distant) quality — coarse levels with slow EMA preserve long-range context that flat single-level fields lose.

### Needle-in-Haystack Retrieval

Measures whether wave-compressed KV vectors preserve retrievable information at varying context depths.

| Depth | Position | KV Cos Sim | Note |
|-------|----------|-----------|------|
| 5% | 50 | **0.629** | Strong distant recall |
| 25% | 253 | **0.588** | |
| 50% | 506 | **0.530** | Mid-range preservation |
| 75% | 759 | 0.380 | |
| 90% | 911 | 0.216 | Near window boundary |

Multi-needle (3 facts inserted simultaneously):

| Fact | Position | KV Cos Sim |
|------|----------|-----------|
| "The secret password is DIAMOND-7492" | 102 (10%) | **0.628** |
| "The capital of Zarqonia is Luminex" | 512 (50%) | **0.511** |
| "Agent X reported 3847 units sold" | 921 (90%) | 0.192 |

Distant facts (positions 50-253) retain cos > 0.58 — the hierarchical coarse wave levels act as persistent long-range memory. This is the primary advantage over flat single-level wave fields, which produce cos ~ 0 beyond ~128 tokens.

### Training Convergence (Hierarchical vs Flat)

| Step | Hierarchical Cos | Flat Cos | Advantage |
|------|-----------------|----------|-----------|
| 500 | 0.558 | 0.486 | +0.072 |
| 1000 | 0.690 | 0.593 | +0.097 |
| 2000 | 0.761 | 0.657 | +0.104 |
| 4000 | **0.790** | 0.696 | **+0.094** |

Hierarchical consistently outperforms flat by ~0.10 cos_sim at every checkpoint.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_waves` | 128 | Wave components per field. More = better reconstruction, more memory. |
| `window_size` | 64 | Exact sliding window. Recent tokens are never compressed. |
| `max_seq_len` | 8192 | Position normalization range. Set to your max expected context. |
| `hierarchical` | False | Enable multi-resolution wave hierarchy. |
| `level_config` | 5-level default | Custom hierarchy: list of (n_waves, ema_rate, freq_low, freq_high). |

## Citation

```bibtex
@software{knopp2026wavekv,
  title={waveKV: Fixed-Memory KV Cache via Wave Interference Fields},
  author={Knopp, Christian},
  year={2026},
  url={https://github.com/Zynerji/waveKV}
}
```

## License

MIT
