from self_snn.agency.self_model import SelfModel, SelfModelConfig


def test_self_key_stability():
    model = SelfModel(SelfModelConfig(key_dim=64))
    key0 = model.key.clone()
    for _ in range(1000):
        model.update_state({"confidence": 0.9, "uncertainty": 0.1}, {"energy": 0.1})
    key1 = model.key
    cos = (key0 * key1).sum() / (key0.norm() * key1.norm())
    assert float(cos) >= 0.95

