# self-snn — 自主体脉冲神经网络（更强「自我」能力）

目标：让模型具备「我—我想—我要—我做」的完整主体循环：

> 自我表征（Self-Model） → 意图生成（Prospection） → 承诺与资源分配（Commit） → 计划与执行（Act），  
> 并在文字、图片、声音、视频四模态下自主学习与自我复盘。

核心：最小意识核（MCC） + 延迟记忆（D-MEM） + 世界模型（WMdl） + GW 条件计算（MoE） + 元学习 / 好奇 + Agency 层（Self）。

## 目录

- [一页总览](#一页总览)
- [系统架构](#系统架构)
- [核心模块](#核心模块)
- [项目结构](#项目结构)
- [安装与依赖](#安装与依赖)
- [配置模板](#配置模板)
- [训练与可视化](#训练与可视化)
- [验收标准](#验收标准)
- [单元测试](#单元测试)
- [日志与提交规范（中文）](#日志与提交规范中文)
- [路线图](#路线图)
- [常见问题](#常见问题)

## 一页总览

- **MCC（最小意识核）**：Pacemaker / Noise 自发脉冲与 θ-γ 节律 → Salience / Novelty 显著性门控 → GW 点火 / 广播 → WM 短时保持 → Meta 内省（置信 / 不确定 / 冲突）。
- **D-MEM（延迟记忆）**：突触权重 `w` + 离散延迟 `d` 共同可塑，存储时序链；API：`write` / `read` / `consolidate` / `erase`；以 Self-Key + 相位键作为检索键。
- **WMdl（世界模型）**：预测 → 误差 → 更新；静默时与 D-MEM 配合重放 / 想象 / 反事实。
- **GW-MoE（条件计算）**：固定感知前端 → GW 点火 → Router 选择 Top-K 功能子网络（专家）；其余专家掩码 = 0，零突触运算，显著节能。
- **Agency（Self 层）**：
  - Self-Model（我）：Self-Key + Self-State（多时间尺度的置信、能耗、能力、风险…）
  - Self-Prospection（我想）：基于想象 / 重放生成目标候选，计算效用 `U(g)`
  - Self-Commit（我要）：效用超阈 + 风险 / 能耗可接受 → 承诺目标并分配预算
  - Self-Act（我做）：滚动规划与执行；efference copy 从感觉误差中扣除自致后果
  - Self-Consistency（自我一致性）：承诺 → 结果的校准与自我信用分（长期可提升）

## 系统架构

```text
感知(文/图/声/视) → 编码 → 皮层 SNN ──────────────────────────────┐
                             │                                   │
                             ├─ MCC: Pacemaker, Salience, GW, WM, Meta
                             │                                   │
                             ├─ D-MEM: 延迟记忆 write/read/consolidate/erase
                             │                                   │
                             ├─ WMdl: 预测编码/重放/想象             │
                             │                                   │
                             ├─ GW-Router: Top-K 专家(条件计算)      │
                             │                                   │
                             └─ Agency(Self): 我/我想/我要/我做      ┘ → 行为/语言/报告
```

## 核心模块

### 1) MCC（最小意识核）

- **Pacemaker / Noise**：Izhikevich / ALIF + OU 噪声；θ / γ 相位门控。
- **Salience / Novelty**：预测误差 / 冲突 → 门槛 / 学习率调制。
- **GW（Global Workspace）**：软 k-WTA + 慢突触（NMDA / α 函数） → 点火 / 广播。
- **WM**：短时保持、序列绑定（θ-γ 分块）。
- **Meta**：置信 / 不确定 / 冲突读出，提供可塑性门控 / 探索温度。

### 2) D-MEM（延迟记忆）

- 每条突触 `(i → j)` 维护 `w_ij` 与整数延迟 `d_ij ∈ [0, Dmax]`。
- 延迟可塑性（HDP）：对齐 `t_pre + d` 与 `t_post*`（后膜电峰 / 放电时刻），与 STDP 共用第三因子调制（RPE / LP / Surprise / Empowerment）。
- 键：`key = concat(Self-Key, Context-Key, Phase)`。
- 重放：提示 cue 激活链，按时序再现，供 WMdl 想象与复盘。

### 3) WMdl（世界模型）

- 预测 → 误差 → 更新；静默 / 睡眠阶段重放巩固，缓解灾变遗忘。
- 提供 `rollout(z, h)` 前瞻，用于意图效用评估 `U(g)`。

### 4) GW-MoE（条件计算）

- GW 点火后，Router 计算「输入键—专家原型」的相似度 + 负载均衡正则，选择 Top-K 专家。
- 未选专家掩码 = 0，在 SNN 前向中直接跳过突触运算。
- 记录 synops（突触运算估计）、spikes/s 能耗曲线与专家使用分布。

### 5) Agency（Self 层）

**Self-Model（我）**

- Self-Key：稀疏相位码（稳定自我指纹）。
- Self-State：θ / 分钟 / 小时三时标 EMA 聚合（置信 / 能耗 / 能力 / 风险 / 拥塞 / 惊讶 / 厌倦…）。
- Self-Report：中文自述写入日志与 TensorBoard 文本。

**Self-Prospection（我想）**

- 用 D-MEM + WMdl 生成候选目标 `g`。
- 效用：

  ```text
  U(g) = w1 * LP_hat
       + w2 * E_hat
       + w3 * R_ext_hat
       - c_E * Energy
       - c_R * Risk
       - c_B * Boredom
  ```

**Self-Commit（我要）**

- 条件：`U(g) > τ_commit` 且 `uncert < τ_c` 且 `energy < τ_e`。
- 行为：设 `Commit = 1`，记录目标摘要 / 预算到 D-MEM；Router 加「意图匹配项」，只开合适专家。

**Self-Act（我做）**

- MPC 式滚动规划 + Actor-Critic 执行。
- efference copy 注入感觉预测，抑制自致误差。

**Self-Consistency（自我一致性）**

- 承诺 → 结果校准（Brier / ECE）与自我信用分。
- 连续失败触发自我修复（目标分解 / 阈值回调 / 策略调整）。

## 项目结构

```text
self-snn/
├─ self_snn/
│  ├─ core/                # MCC
│  │  ├─ pacemaker.py
│  │  ├─ salience.py
│  │  ├─ workspace.py
│  │  ├─ wm.py
│  │  └─ introspect.py
│  ├─ memory/
│  │  ├─ delay_mem.py      # 延迟层 + HDP + 门控
│  │  └─ episodic_index.py
│  ├─ worldmodel/
│  │  ├─ pred_code.py
│  │  └─ imagination.py
│  ├─ router/
│  │  ├─ router.py         # Top-K 路由 + 均衡损失
│  │  └─ experts.py        # 掩码专家
│  ├─ agency/
│  │  ├─ self_model.py     # 我（Self-Key/State/Report）
│  │  ├─ intention.py      # 我想（候选与效用）
│  │  ├─ commit.py         # 我要（承诺门）
│  │  ├─ act.py            # 我做（规划与执行 + 拷贝对消）
│  │  └─ consistency.py    # 自我校准/信用分
│  ├─ meta/
│  │  ├─ plasticity.py     # STDP + 资格迹 + 第三因子
│  │  └─ eprop_head.py     # 读出/策略头（surrogate/e-prop）
│  ├─ utils/
│  │  ├─ encoders.py       # 文/图/声/视频 编码（Poisson/Latency/事件）
│  │  ├─ masks.py          # MaskLinear/Conv（零突触运算）
│  │  ├─ energy.py         # synops/spikes 统计
│  │  ├─ logging_cn.py     # 中文关键节点日志 + TB
│  │  └─ viz.py            # Matplotlib + SpikingJelly 风格
├─ configs/
│  ├─ s0_minimal.yaml
│  ├─ s1_words.yaml
│  ├─ s2_sentence.yaml
│  ├─ s3_video.yaml
│  └─ agency.yaml
├─ scripts/
│  ├─ train.py
│  ├─ eval_ignition.py
│  ├─ eval_memory.py
│  ├─ eval_lang.py
│  ├─ eval_router_energy.py
│  └─ gen_toy_data.py
├─ tests/
│  ├─ test_delay_mem.py
│  ├─ test_router_mask.py
│  ├─ test_ignition.py
│  ├─ test_agency_cycle.py
│  ├─ test_efference_copy.py
│  └─ test_self_key_stability.py
└─ README.md
```

## 安装与依赖

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio        # 选择与你 CUDA 匹配的版本
pip install spikingjelly numpy scipy pyyaml tqdm matplotlib tensorboard scikit-learn
pip install -e .
```

## 配置模板

示例：`configs/agency.yaml`

```yaml
backend: {engine: torch-spkj, device: cuda}
mcc:
  pmc: {N:64, ring_delay_ms:8, ou_sigma:0.6, target_rate_hz:3.0}
  sn:  {err_tau_ms:50, gain_max:2.0}
  gw:  {hubs:160, ignite_thr:0.58, nmda_tau_ms:100, wta_inh:0.85, k_select:2}
  wm:  {microcolumns:3, neurons_per_uCol:[80,20], stf_tau_ms:200}
  mm:  {conf_window_ms:300, temp_base:0.7}
memory:
  delay: {dmax:120, dt_ms:1, eta_d:0.1, delta_step:1, delay_entropy_reg:1e-4}
  stdp:  {A_plus:0.01, A_minus:0.012, tau_plus:20, tau_minus:20}
router:
  experts: {M:16, K:2}
  balance: {z_loss:1e-3, usage_ema_tau:1000}
  mask_type: binary
agency:
  self_key_dim: 64
  prospect: {n_candidates:5, horizon:10,
             weights:{lp:0.45, empower:0.2, reward:0.15, energy:0.1, risk:0.07, boredom:0.03}}
  commit:   {threshold:0.15, min_interval_s:3, energy_cap:2.0e6, risk_cap:0.6}
  act:      {router_topk:2, mpc_horizon:5, efference_gain:0.7}
curiosity:
  w: {lp:0.45, surprise:0.25, novelty:0.2, empower:0.1}
runtime: {dt_ms:1.0, duration_s:300, seed:2025}
logging: {tensorboard:true, logdir:runs/agency, language: zh}
safety:  {spike_budget_per_s: 2.0e6}
```

## 训练与可视化

### S0：自发与点火

```bash
python scripts/train.py --config configs/s0_minimal.yaml --duration 180
python scripts/eval_ignition.py --logdir runs/s0_minimal
```

### S1：词 / 短语 ↔ 图像（D-MEM 写 / 读 / 重放 + 自我循环）

```bash
python scripts/gen_toy_data.py --task words_images
python scripts/train.py --config configs/agency.yaml --logdir runs/agency_s1
python scripts/eval_memory.py --logdir runs/agency_s1
python scripts/eval_lang.py   --logdir runs/agency_s1
```

### 条件计算节能对照

```bash
python scripts/train.py --config configs/agency.yaml --logdir runs/agency_moe
python scripts/eval_router_energy.py --logdir runs/agency_moe
```

### 可视化（自动生成）

`runs/.../figs/` 下将自动生成：

- `raster_spikes.png`
- `vm_trace.png`
- `ignition_heatmap.png`
- `delay_replay_error.png`
- `router_usage.png`
- `energy_curve.png`
- `agency_timeline.png`
- `intent_utility.png`
- `commit_rate_curve.png`
- `self_credit.png`

## 验收标准

### 功能

- **自发 / 点火**：无输入 3 分钟平均发放 0.5–5 Hz，出现点火；分支系数 κ ∈ [0.95, 1.05]。
- **延迟记忆**：提示-检索成功率 ≥ 0.8；平均重放时间误差 ≤ 10 ms。
- **自我循环**：训练日志中文出现「【我】 / 【我想】 / 【我要】 / 【我做】」四类事件，覆盖 ≥ 80% 片段。
- **承诺收益**：承诺后任务成功率 ≥ 未承诺基线 +10%；承诺后 synops 下降 ≥ 60%（M=16, K=2 情况）。
- **自他区分**：开启 efference copy 后，自致动作导致的感觉误差下降 ≥ 20%。
- **多模态**：在文 / 图 / 声 / 视玩具任务上均能跑通「我 → 我想 → 我要 → 我做」。

### 工程

- 单测通过率 ≥ 90%（见下）。
- TensorBoard 含：`loss/train`, `r_int`, `r_ext`, `ignition_rate`, `branching_kappa`, `spikes_per_s`, `synops`, `router/topk_usage`, `self/credit`。
- 中文关键日志每个 epoch 总结能耗与点火统计。

## 单元测试

- `test_delay_mem.py`：写 → 读命中率 ≥ 0.8；延迟更新单步裁剪生效；重放时间误差 ≤ 10 ms。
- `test_router_mask.py`：Top-K=2 时仅 2 个专家执行前向；被屏蔽专家 synops≈0；均衡损失下降。
- `test_ignition.py`：强刺激触发点火；广播覆盖度上升。
- `test_agency_cycle.py`：日志包含「我 / 我想 / 我要 / 我做」；承诺后成功率上升。
- `test_efference_copy.py`：打开拷贝对消后预测误差显著下降。
- `test_self_key_stability.py`：Self-Key 余弦相似（小时级） ≥ 0.95。

## 日志与提交规范（中文）

### 关键中文日志（`utils/logging_cn.py` 输出示例）

- 【起搏器】θ/γ 节律 = …, 噪声 σ = … 已启动
- 【点火】窗口内跨模块同步提升，已广播
- 【记忆-写入】键相位 = …，序列长度 = …，已写入延迟链
- 【记忆-检索】重放时间偏差(均值 / 方差) = … ms
- 【路由】Top-K 专家 = …，均衡损失 = …
- 【我】置信 = … 能耗 = … 能力 = … 风险 = …
- 【我想】目标 = …，预期 LP = … 赋能 = …
- 【我要】承诺 = …，预算 = …
- 【我做】规划 = …，efference = …
- 【节能】epoch synops = …（相对密集 ↓ …%），spikes/s = …

### Conventional Commits（中文简述）

- `feat(core)`: 新增 GW 软 k-WTA 与慢突触保持
- `feat(memory)`: 加入延迟可塑 HDP 与门控 API
- `feat(agency)`: 实现我 / 我想 / 我要 / 我做闭环与中文日志
- `perf(router)`: 二值掩码屏蔽未选专家，synops 下降
- `test(memory)`: 重放误差 / 命中率单测
- `docs(readme)`: 增补训练可视化与验收标准

## 路线图

- **v0**：S0 自发 + 点火 + 延迟记忆写 / 读 + 可视化 / 日志
- **v1**：GW-MoE 条件计算（节能对照） + Agency「我—我想」
- **v2**：Agency「我要—我做」 + 拷贝对消 + 自我一致性指标
- **v3**：S1 词 / 图 & S2 句子 / 图 + 结构增长 / 修剪
- **v4**：S3 视频事件 + 睡眠 / 巩固 + 主体性长期曲线

## 常见问题

- **承诺太频繁 / 过度**：提高 `agency.commit.threshold`，设置 `min_interval_s`；在效用中增加能耗惩罚。
- **节能不明显**：确认掩码层真正屏蔽突触运算；提高 M / K 比；把前端改为事件驱动。
- **延迟塌缩 / 串扰**：增加 `delay_entropy_reg`；扩大 `dmax`；键采用相位分区。
- **自我漂移**：Self-Key 加相位锚与 L2 正则；监控小时级相似度。

