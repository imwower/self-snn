import torch

from self_snn.worldmodel.pred_code import PredictiveCoder, PredictiveCoderConfig


def test_predictive_coder_rollout_shapes_and_training():
    """基础预测编码器：rollout 形状正确，简单训练后误差应下降。"""
    torch.manual_seed(42)
    cfg = PredictiveCoderConfig(hidden_dim=16)
    model = PredictiveCoder(cfg)

    # 构造一个固定目标向量 z
    z = torch.randn(16)

    # 初始误差
    with torch.no_grad():
        pred0, err0 = model.forward(z)
        init_l2 = float((err0**2).mean())

    # 简单训练若干步，最小化 L2 误差
    opt = torch.optim.SGD(model.net.parameters(), lr=0.1)
    for _ in range(20):
        opt.zero_grad()
        pred, err = model.forward(z)
        loss = (err**2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred1, err1 = model.forward(z)
        final_l2 = float((err1**2).mean())

    # 误差应有明显下降
    assert final_l2 < init_l2

    # 检查 rollout 形状
    out = model.rollout(z, horizon=5)
    traj = out["traj"]
    err_traj = out["err"]
    assert traj.shape[0] == 6  # T+1
    assert traj.shape[1] == cfg.hidden_dim
    assert err_traj.shape[0] == 5
    assert err_traj.shape[1] == cfg.hidden_dim

