import torch

from self_snn.router.router import GWRouter, RouterConfig


def test_topk_router_mask():
    cfg = RouterConfig(num_experts=16, k=2)
    router = GWRouter(cfg)
    wm_state = torch.randn(128)
    mask, stats = router(wm_state)
    assert int(mask.sum()) == cfg.k
    assert stats["probs"].shape[0] == cfg.num_experts

