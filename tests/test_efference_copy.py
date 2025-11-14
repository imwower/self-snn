import torch

from self_snn.agency.act import ActModule, ActConfig


def test_efference_reduces_error():
    act = ActModule(ActConfig(efference_gain=0.7))
    wm_state = torch.ones(16)
    commit_state = {"committed": True, "goal": torch.ones(6)}
    gw_mask = torch.ones(16)
    out = act(commit_state, wm_state, gw_mask)
    err_without = wm_state.mean()
    err_with = wm_state.mean() - out["efference"]
    assert abs(err_with) < abs(err_without)

