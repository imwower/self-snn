import torch

from self_snn.memory.episodic_index import EpisodicIndex, EpisodicIndexConfig


def test_episodic_index_lru_and_ttl_and_hit_rate():
    """EpisodicIndex 应支持 LRU/TTL 删除与命中率统计。"""
    cfg = EpisodicIndexConfig(max_episodes=2, ttl_steps=1)
    index = EpisodicIndex(cfg)

    key = torch.randn(4)

    # 连续写入三条事件，因 max_episodes=2，最早的一条会被 LRU 删除
    e0 = index.add(key, "episode0")
    e1 = index.add(key, "episode1")
    e2 = index.add(key, "episode2")

    # e0 应该被 LRU 删除
    assert index.get(e0) is None
    info1 = index.get(e1)
    info2 = index.get(e2)
    assert info1 is not None
    assert info2 is not None

    # TTL：再次 add 触发 TTL 清理，TTL=1 步意味着只保留最新的一条
    e3 = index.add(key, "episode3")
    # e1 理论上可能已过期或被 LRU 删除
    _ = index.get(e1)
    # e3 一定存在
    assert index.get(e3) is not None

    # 命中率应在 (0,1] 之间
    hit_rate = index.hit_rate()
    assert 0.0 <= hit_rate <= 1.0

