from self_snn.core.workspace import SelfSNN, SelfSNNConfig


def test_ignition_rate_reasonable():
    model = SelfSNN(SelfSNNConfig())
    out = model(steps=300)
    spikes = out["spikes"]
    rate_hz = float(spikes.float().mean() * 1000.0)
    assert 0.5 <= rate_hz <= 5.0

