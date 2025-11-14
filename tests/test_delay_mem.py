import torch

from self_snn.memory.delay_mem import DelayMemory, DelayMemoryConfig


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

