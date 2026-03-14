"""Tests for waveKV."""

import torch
import pytest
from wave_kv import WaveKVManager, WaveKVCache, WaveFieldKV


class TestWaveFieldKV:
    def test_init(self):
        field = WaveFieldKV(head_dim=64, n_waves=32)
        assert field.n_waves == 32
        assert field.head_dim == 64

    def test_update_and_reconstruct(self):
        field = WaveFieldKV(head_dim=64, n_waves=32)
        kv = torch.randn(64)
        field.update(kv, position=0)
        positions = torch.tensor([0])
        out = field.reconstruct(positions)
        assert out.shape == (1, 64)

    def test_batch_update(self):
        field = WaveFieldKV(head_dim=64, n_waves=32)
        kv = torch.randn(10, 64)
        positions = torch.arange(10)
        field.update_batch(kv, positions)
        assert field.n_updates.item() == 10

    def test_reset(self):
        field = WaveFieldKV(head_dim=64, n_waves=32)
        field.update(torch.randn(64), 0)
        field.reset()
        assert field.n_updates.item() == 0
        assert field.amps.sum().item() == 0.0


class TestWaveKVCache:
    def test_init(self):
        cache = WaveKVCache(n_layers=2, n_heads=4, head_dim=32, n_waves=16, window_size=8)
        assert cache.current_pos == 0

    def test_update_and_reconstruct_within_window(self):
        cache = WaveKVCache(n_layers=1, n_heads=2, head_dim=32, n_waves=16, window_size=8)
        k = torch.randn(2, 4, 32)  # [heads, seq, dim]
        v = torch.randn(2, 4, 32)
        positions = torch.arange(4)
        cache.update(0, k, v, positions)
        rk, rv = cache.reconstruct(0)
        assert rk.shape == (2, 4, 32)
        assert rv.shape == (2, 4, 32)

    def test_memory_bytes(self):
        cache = WaveKVCache(n_layers=2, n_heads=4, head_dim=32, n_waves=16, window_size=8)
        assert cache.memory_bytes > 0
        assert cache.equivalent_standard_bytes == 0  # no tokens cached yet

    def test_reset(self):
        cache = WaveKVCache(n_layers=1, n_heads=2, head_dim=32, n_waves=16, window_size=8)
        k = torch.randn(2, 4, 32)
        v = torch.randn(2, 4, 32)
        cache.update(0, k, v, torch.arange(4))
        cache.reset()
        assert cache.current_pos == 0


class TestWaveKVManager:
    def test_init(self):
        mgr = WaveKVManager(n_layers=2, n_heads=4, head_dim=32, device="cpu")
        assert mgr.seq_len == 0

    def test_get_kv_empty(self):
        mgr = WaveKVManager(n_layers=2, n_heads=4, head_dim=32, device="cpu")
        assert mgr.get_kv(0) is None

    def test_absorb_and_get(self):
        mgr = WaveKVManager(n_layers=1, n_heads=2, head_dim=32,
                            n_waves=16, window_size=8, device="cpu")
        # Simulate attention output: [batch=1, heads=2, seq=4, dim=32]
        new_kv = (torch.randn(1, 2, 4, 32), torch.randn(1, 2, 4, 32))
        mgr.absorb(0, new_kv, n_new=4)
        mgr.step()
        assert mgr.seq_len == 4
        kv = mgr.get_kv(0)
        assert kv is not None
        k, v = kv
        assert k.shape[0] == 1  # batch
        assert k.shape[1] == 2  # heads

    def test_stats(self):
        mgr = WaveKVManager(n_layers=2, n_heads=4, head_dim=32,
                            n_waves=16, window_size=8, device="cpu")
        new_kv = (torch.randn(1, 4, 4, 32), torch.randn(1, 4, 4, 32))
        mgr.absorb(0, new_kv, n_new=4)
        mgr.step()
        stats = mgr.stats()
        assert stats["cached_tokens"] == 4
        assert stats["wave_cache_mb"] > 0

    def test_reset(self):
        mgr = WaveKVManager(n_layers=1, n_heads=2, head_dim=32, device="cpu")
        new_kv = (torch.randn(1, 2, 4, 32), torch.randn(1, 2, 4, 32))
        mgr.absorb(0, new_kv, n_new=4)
        mgr.step()
        mgr.reset()
        assert mgr.seq_len == 0
        assert mgr.get_kv(0) is None
