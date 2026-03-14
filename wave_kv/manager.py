"""WaveKVManager — drop-in KV cache replacement for any transformer."""

import torch
import torch.nn as nn

from wave_kv.cache import WaveKVCache


class WaveKVManager:
    """Drop-in wave KV cache manager for any transformer.

    Produces standard (k, v) tuples that any attention layer expects.
    Memory is fixed regardless of sequence length.

    Args:
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        n_waves: Wave components per field.
        window_size: Exact sliding window for recent tokens.
        max_seq_len: Maximum expected sequence length.
        device: Device for cache tensors.
    """

    def __init__(self, n_layers, n_heads, head_dim, n_waves=128,
                 window_size=64, max_seq_len=8192, device="cuda",
                 hierarchical=False, level_config=None):
        if isinstance(device, str):
            device = torch.device(device)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.cache = WaveKVCache(
            n_layers=n_layers, n_heads=n_heads, head_dim=head_dim,
            n_waves=n_waves, window_size=window_size, max_seq_len=max_seq_len,
            hierarchical=hierarchical, level_config=level_config,
        ).to(device)
        self._total_pos = 0
        self._step_tokens = 0

    def get_kv(self, layer_idx):
        """Get cached (k, v) for attention. Returns None if empty."""
        if self._total_pos == 0:
            return None
        k, v = self.cache.reconstruct(layer_idx)
        return k.unsqueeze(0), v.unsqueeze(0)

    def absorb(self, layer_idx, new_kv, n_new):
        """Store new K,V from attention output. Extracts last n_new tokens."""
        full_k, full_v = new_kv
        k_new = full_k[0, :, -n_new:, :]
        v_new = full_v[0, :, -n_new:, :]
        positions = torch.arange(
            self._total_pos, self._total_pos + n_new, device=self.device)
        self.cache.update(layer_idx, k_new, v_new, positions)
        self._step_tokens = n_new

    def step(self):
        """Advance position counter. Call once after all layers processed."""
        self._total_pos += self._step_tokens
        self._step_tokens = 0

    def reset(self):
        """Clear all cached state."""
        self.cache.reset()
        self._total_pos = 0
        self._step_tokens = 0

    @property
    def seq_len(self):
        return self._total_pos

    def stats(self):
        """Memory usage statistics."""
        wave_mb = self.cache.memory_bytes / 1e6
        std_mb = self.cache.equivalent_standard_bytes / 1e6
        ratio = std_mb / wave_mb if wave_mb > 0 else 0
        return {
            "cached_tokens": self._total_pos,
            "wave_cache_mb": round(wave_mb, 2),
            "standard_cache_mb": round(std_mb, 2),
            "compression_ratio": round(ratio, 1),
        }

    @classmethod
    def for_model(cls, model, n_waves=128, window_size=64, max_seq_len=8192):
        """Auto-create from any transformer model (HF, nanoGPT, LLaMA, etc.)."""
        config = getattr(model, "config", None)
        n_layers = n_heads = head_dim = None

        if config:
            for a in ["num_hidden_layers", "n_layer", "num_layers"]:
                n_layers = n_layers or getattr(config, a, None)
            for a in ["num_key_value_heads", "num_attention_heads", "n_head"]:
                n_heads = n_heads or getattr(config, a, None)
            hidden = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
            if n_layers and n_heads and hidden:
                head_dim = hidden // (getattr(config, "num_attention_heads", None) or n_heads)
                device = next(model.parameters()).device
                return cls(n_layers, n_heads, head_dim, n_waves, window_size, max_seq_len, device)

        for attr in ["layers", "blocks", "h"]:
            layers = getattr(model, attr, None)
            if layers is None and hasattr(model, "model"):
                layers = getattr(model.model, attr, None)
            if layers is not None and hasattr(layers, "__len__"):
                n_layers = len(layers)
                first = layers[0]
                attn = getattr(first, "attn", getattr(first, "self_attn", None))
                if attn:
                    n_heads = getattr(attn, "n_head", getattr(attn, "num_heads", None))
                    head_dim = getattr(attn, "head_dim", None)
                    if not head_dim and n_heads:
                        h = getattr(attn, "n_embd", getattr(attn, "hidden_size", None))
                        if h:
                            head_dim = h // n_heads
                if n_layers and n_heads and head_dim:
                    device = next(model.parameters()).device
                    return cls(n_layers, n_heads, head_dim, n_waves, window_size, max_seq_len, device)

        raise ValueError(
            "Cannot auto-detect architecture. "
            "Use WaveKVManager(n_layers=..., n_heads=..., head_dim=...) instead."
        )
