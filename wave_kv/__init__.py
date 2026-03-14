"""waveKV — Drop-in wave-encoded KV cache for any transformer.

Fixed O(1) memory regardless of sequence length.
Pure PyTorch. Works on NVIDIA, AMD, Intel, Apple Silicon, CPU.

Usage:
    from wave_kv import WaveKVManager

    cache = WaveKVManager.for_model(model)
    # or: cache = WaveKVManager(n_layers=32, n_heads=32, head_dim=128)
"""

from wave_kv.cache import WaveKVCache, WaveFieldKV, HierarchicalWaveFieldKV
from wave_kv.manager import WaveKVManager

__version__ = "0.2.0"
__all__ = ["WaveKVCache", "WaveKVManager", "WaveFieldKV", "HierarchicalWaveFieldKV"]
