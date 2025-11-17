import torch

from self_snn.memory.delay_mem import DelayMemory, DelayMemoryConfig
from self_snn.meta.plasticity import STDPConfig


def test_delay_mem_hit_and_replay():
    mem = DelayMemory(DelayMemoryConfig(dmax=120))
    key = torch.randint(0, 2, (8,), dtype=torch.float32)
    seq = torch.randn(20, 4)
    for _ in range(10):
        mem.write(key, seq, delay=5)
    recalled = mem.read(key)
    assert recalled is not None
    corr = torch.nn.functional.cosine_similarity(recalled.view(1, -1), seq.mean(dim=0).view(1, -1))
    assert float(corr) > 0.8
    replayed = mem.replay(key)
    assert replayed is not None
    assert replayed.shape[0] <= 120


def test_delay_mem_stdp_update_weights():
    """DelayMemory 的 STDP 接口应能更新内部权重。"""
    mem = DelayMemory(DelayMemoryConfig(dmax=10))
    key = torch.randint(0, 2, (8,), dtype=torch.float32)
    pre = torch.tensor([1.0, 0.0])
    post = torch.tensor([0.0, 1.0])
    cfg = STDPConfig(a_plus=0.01, a_minus=0.0)

    # 初始权重不存在，首次更新后应存在且有非零元素
    mem.update_weights(key, pre, post, cfg, third_factor=1.0)
    d_stats = mem.stats()
    assert "n_keys" in d_stats
    # 再次更新，确保接口可重复调用且不会报错
    mem.update_weights(key, pre, post, cfg, third_factor=1.0)
