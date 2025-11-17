import torch

from self_snn.meta.plasticity import STDPConfig, STDPState, stdp_3factor, stdp_update


def test_stdp_update_basic_increase():
    """简单两因子 STDP：pre/post 同步发放时权重应发生变化。"""
    w = torch.zeros(2, 2)
    pre = torch.tensor([1.0, 0.0])
    post = torch.tensor([0.0, 1.0])
    cfg = STDPConfig(a_plus=0.01, a_minus=0.0)
    w_new = stdp_update(w, pre, post, cfg)
    # 至少有一个权重被正向更新
    assert torch.any(w_new > 0)


def test_stdp_3factor_third_factor_scales_update():
    """第三因子为 0 时不更新，为正时更新幅度随之缩放。"""
    w = torch.zeros(2, 2)
    pre = torch.tensor([1.0, 0.0])
    post = torch.tensor([0.0, 1.0])
    cfg = STDPConfig(a_plus=0.01, a_minus=0.0)
    state = STDPState(pre_trace=torch.zeros_like(pre), post_trace=torch.zeros_like(post))

    # 第三因子为 0：权重不变
    w_zero, state_zero = stdp_3factor(w, pre, post, state, cfg, third_factor=0.0)
    assert torch.allclose(w_zero, w)

    # 第三因子为 1：权重有正增量
    w_one, _ = stdp_3factor(w, pre, post, state_zero, cfg, third_factor=1.0)
    assert torch.any(w_one > 0)


def test_stdp_eligibility_trace_decay():
    """资格迹在无新脉冲时应随时间衰减。"""
    pre = torch.ones(3)
    post = torch.ones(2)
    cfg = STDPConfig(a_plus=0.01, a_minus=0.01, tau_plus=20.0, tau_minus=20.0)
    state = STDPState(pre_trace=torch.ones_like(pre), post_trace=torch.ones_like(post))

    w = torch.zeros(3, 2)

    # 第一步：有脉冲输入，更新资格迹
    w1, state1 = stdp_3factor(w, pre, post, state, cfg, third_factor=1.0)
    # 第二步：无脉冲，仅凭衰减
    pre_zero = torch.zeros_like(pre)
    post_zero = torch.zeros_like(post)
    _, state2 = stdp_3factor(w1, pre_zero, post_zero, state1, cfg, third_factor=0.0)

    assert torch.all(state2.pre_trace < state1.pre_trace)
    assert torch.all(state2.post_trace < state1.post_trace)

