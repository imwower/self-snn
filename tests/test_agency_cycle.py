from self_snn.core.workspace import SelfSNN, SelfSNNConfig
from self_snn.utils.logging_cn import setup_logger, log_self, log_self_think, log_self_want, log_self_do


def test_agency_cycle_logging(tmp_path):
    model = SelfSNN(SelfSNNConfig())
    out = model(steps=100)
    logger = setup_logger(tmp_path)
    log_self(logger, "置信=..., 能耗=..., 能力=..., 风险=...")
    log_self_think(logger, "目标=..., 预期LP=..., 赋能=...")
    log_self_want(logger, "承诺=..., 预算=...")
    log_self_do(logger, "规划=..., efference=...")
    assert "commit_state" in out

