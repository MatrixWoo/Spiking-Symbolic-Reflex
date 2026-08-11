# SNN Reflex Demo — 代码设计讲解

> 面向导师的完整技术说明，涵盖架构设计、模块实现、实验结果和后续规划。

---

## 1. 你要解决什么问题？

传统机器人控制是 **感知 → 规划 → 执行** 的串行流水线。问题在于：当抓取过程中发生意外滑移，这条链路太慢了——等感知层处理完、规划层重新计算、再发指令，物体已经掉了。

核心 idea 是：**在常规控制回路之外，加一条 Event → Concept → Rule → Reflex Action 的快速旁路**，用 SNN 的事件驱动特性实现低延迟反射。

---

## 2. 整体架构

```
                 ┌──────────────────────────────┐
  Environment    │  Event Detector              │
  ───────────►   │  (event_detector.py)         │
  状态向量        │  distance_increase ──┐       │
  [gripper_x,    │  velocity_anomaly ───┤       │
   gripper_z,    │  contact_loss ───────┤       │
   object_x,     │  slip_risk ──────────┤       │
   object_z,     │  grasp_unstable ─────┘       │
   ...]          └──────────┬───────────────────┘
                            │ events [0,1]
                            ▼
                 ┌──────────────────────────────┐
                 │  Concept Layer               │
                 │  (concept_layer.py)          │
                 │                              │
                 │  events → concepts           │
                 │  slip, grasp_unstable,       │
                 │  object_falling,             │
                 │  recovery_needed             │
                 └──────┬───────────┬───────────┘
                        │ concepts  │ events
                        ▼            ▼
           ┌──────────────┐  ┌──────────────────┐
           │  Rule Layer  │  │  SNN Reflex      │
           │  (rule_layer │  │  (snn_reflex.py) │
           │   .py)       │  │                  │
           │              │  │  5→10→2 LIF      │
           │  IF slip     │  │  event spikes →  │
           │  THEN tighten│  │  reflex signal   │
           └──────┬───────┘  └────────┬─────────┘
                  │ reflex_action     │ snn_signal
                  ▼                   ▼
           ┌─────────────────────────────────────┐
           │  Environment.step(action, reflex)    │
           │  Reflex overrides planned action     │
           └─────────────────────────────────────┘
```

**设计要点：Rule Layer 和 SNN 是并行互补的，不是串行。** Rule 提供可解释的符号决策，SNN 提供事件驱动的快速响应。

---

## 3. 逐模块讲解

### 3.1 Environment (`environment.py`)

**设计原则：够用即可，不追求物理精度。**

```python
@dataclass
class EnvConfig:
    dt: float = 0.01          # 10ms 仿真步长
    slip_time: float = 2.5    # 滑移触发时间
    slip_magnitude: float = 8.0  # 滑移力 > 正常摩擦力(2.75N)
    gripper_close_force: float = 5.0   # 正常抓取力
    gripper_tighten_force: float = 18.0 # 反射增强力
```

核心物理只有三个方程：

1. **抓取状态**：`grasped = contact AND width < threshold` — 夹爪合拢到阈值以下且接触物体
2. **滑移条件**：`slip_force > friction_force = grip_force × μ` — 滑移力超过最大静摩擦力时物体开始滑动
3. **自由落体**：`vz -= g·dt`，碰撞地面后回弹衰减

**关键设计决策**：`reflex_applied` 是一次性标志位。反射只触发一次，模拟真实的快速反射——你不会对同一个扰动反复反射。

```python
if reflex_action is not None and not self.reflex_applied:
    self.reflex_applied = True
    self.reflex_time = self.time
    if reflex_action == 6:  # tighten
        self.gripper_force = cfg.gripper_tighten_force  # 5N → 18N
```

### 3.2 Event Detector (`event_detector.py`)

**设计原则：事件应该是连续的激活值 [0,1]，不是离散的 0/1。这样 SNN 可以接收模拟信号。**

5 种事件，每种都是归一化到 [0,1] 的连续值：

| 事件 | 物理含义 | 计算方式 |
|------|---------|---------|
| `distance_increase` | 物体远离夹爪 | `clip(Δdist / 0.05, 0, 1)` |
| `velocity_anomaly` | 物体速度异常 | `clip((speed-0.05) / 0.3, 0, 1)` |
| `contact_loss` | 接触丢失 | 接触状态变化时的 0→1 跳变 |
| `slip_risk` | 综合滑移风险 | `0.6×dist + 0.4×velocity` |
| `grasp_unstable` | 抓取不稳定 | 接触状态下速度超阈值 |

关键设计：事件检测是 **state-difference** 驱动的，不是帧差。`detect(curr_state, prev_state, env)` 对比前后两帧状态，不依赖任何学习参数。

### 3.3 Concept Layer (`concept_layer.py`)

**设计原则：这是 neural → symbolic 的桥梁。连续事件 → 符号概念。**

```python
# 核心映射
raw['slip'] = slip_risk*0.5 + distance_increase*0.3 + velocity_anomaly*0.2
raw['grasp_unstable'] = grasp_unstable*0.6 + slip_risk*0.4
raw['object_falling'] = contact_loss*0.7 + velocity_anomaly*0.3
raw['recovery_needed'] = max(slip*0.6, grasp_unstable*0.5, object_falling*0.8)
```

每个概念加了**时间平滑**（指数移动平均），防止单帧噪声误触发：

```python
concept = α × raw + (1-α) × prev_concept
```

这模拟了生物神经系统中突触整合的时间常数。

### 3.4 Rule Layer (`rule_layer.py`)

**设计原则：可解释、可审计的符号规则。4 条规则，fuzzy matching。**

```python
rules = [
    {'name': 'tighten_for_slip',
     'conditions': {'slip': 0.3, 'grasp_unstable': 0.3},
     'action': 6,   # tighten_grip
     'weight': 1.0},
    {'name': 'regrasp_on_fall',
     'conditions': {'object_falling': 0.5},
     'action': 7,   # regrasp
     'weight': 1.2},
    {'name': 'regrasp_on_contact_loss',
     'conditions': {'contact_loss': 0.5},
     'action': 7,   # regrasp
     'weight': 1.5},
    {'name': 'tighten_on_recovery',
     'conditions': {'recovery_needed': 0.4, 'slip': 0.2},
     'action': 6,   # tighten_grip
     'weight': 0.8},
]
```

**Fuzzy AND**：用 `min()` 组合多个条件，保证只有所有条件都满足时置信度才高。每条规则有权重，最终选最高置信度的规则触发。

### 3.5 SNN Reflex Module (`snn_reflex.py`) — 核心模块

**设计原则：从零实现 LIF 神经元，不依赖 snntorch/brian2。这是为了让你真正理解脉冲计算。**

#### LIF 动力学

```python
dv/dt = (-(v - v_rest) + I × gain) / tau
if v >= v_thresh:
    spike!
    v = v_reset
```

三个关键参数：

- `tau = 8ms`：快膜时间常数 → 快速响应
- `gain = 8–12`：高增益 → 对弱事件也敏感
- `v_thresh = -55mV, v_rest = -70mV`：阈值 15mV 间距 → 快速积累到阈值

#### 网络结构：5 → 10 → 2

```
输入层 (5个事件值，非脉冲神经元，直接传入)
   │ weights: 10×5 矩阵, 均值~3, 85%兴奋性
   ▼
隐藏层 (10个 LIF 神经元, tau=8-16ms, gain=8-12)
   │ weights: 2×10 矩阵, 均值~5
   ▼
输出层 (2个 LIF 神经元, tau=8ms, gain=8)
   → [tighten_grip 置信度, regrasp 置信度]
```

**为什么用 10 个隐藏神经元？** 够少——这不是在追求分类精度，而是展示脉冲编码的基本能力。10 个神经元的不同调谐特性（不同 tau、gain 的随机分布）形成了对事件模式的分布式表示。

#### 脉冲→决策转换

```python
avg_spikes = 滑动窗口平均(最近20帧的输出脉冲)
tighten_conf = clip(avg_spikes[0] × 10.0, 0, 1)
```

用滑动窗口平滑输出脉冲，避免单帧噪声。×10 的缩放因子是因为脉冲本身是稀疏的（大部分时间步没有脉冲）。

---

## 4. 实验设计

三个实验形成消融对比：

| 实验 | 反射机制 | 预期 | 实际 |
|------|---------|------|------|
| No Reflex | 无 | 物体掉落 | ✗ 掉落 (t=2.75s) |
| Rule Reflex | 仅规则 | 恢复成功 | ✓ 恢复 (30ms) |
| SNN + Rule | 规则+SNN | 恢复成功，SNN 提供冗余确认 | ✓ 恢复 (30ms, 305 脉冲) |

**为什么 SNN+Rule 和 Rule-Only 延迟相同？** 因为 demo 里 Rule 本身已经很快（30ms），SNN 的贡献体现在：

1. **冗余确认**：73 次 reflex response 持续确认规则决策
2. **稀疏性**：305 脉冲 / (500步 × 12 神经元) = 5% 活跃度
3. **如果规则失效**（比如规则写得不全），SNN 可以直接触发反射

**关键设计**：反射只在运输阶段启用（`was_grasped AND gripper_z > 0.08`），防止抓取阶段的假阳性触发。

---

## 5. 这个 demo 证明了什么

1. **闭环成立**：Event → Concept → Rule → Reflex Action 链路可以闭合
2. **SNN 可以工作**：LIF 神经元对滑移事件产生可靠的脉冲响应
3. **规则+SNN 互补**：规则提供可解释性，SNN 提供事件驱动的快速确认
4. **框架可扩展**：事件类型、概念数量、规则数量、SNN 层数都可以增加

---

## 6. 接下来可以做什么

| 方向 | 内容 |
|------|------|
| **差异化 SNN 延迟** | 给 Rule 加 50ms 延迟（模拟规划时间），对比 SNN 的 10ms 响应 |
| **更多扰动类型** | 碰撞、风扰、负载变化 |
| **SNN 学习** | 用 STDP 替代随机权重，让 SNN 从经验中学习反射模式 |
| **迁移到 PyBullet** | 保留完全相同的 Event/Concept/Rule/SNN 接口 |
| **多模态事件** | 加入触觉传感器 (force/torque) 作为额外事件源 |

---

## 7. 文件结构

```
plaining/
├── main.py              # 主入口，配置参数，运行实验
├── environment.py       # 玩具环境：2D 夹爪+物体+滑移物理
├── event_detector.py    # 事件检测：5种事件，state-difference 驱动
├── concept_layer.py     # 概念层：连续事件→符号概念 + 时间平滑
├── rule_layer.py        # 规则层：4条 fuzzy 符号规则
├── snn_reflex.py        # SNN 模块：5→10→2 LIF 脉冲网络
├── visualization.py     # 可视化：6张图，消融对比
├── idea/                # 设计文档
│   └── design-notes.md  # 本文档
└── output/              # 生成图表
    ├── experiment_no_reflex.png
    ├── experiment_rule_reflex.png
    ├── experiment_snn_rule_reflex.png
    ├── comparison.png
    ├── metrics_summary.png
    └── snn_spike_raster.png
```

---

## 8. 运行方式

```bash
conda activate snn-demo
cd /home/wuzuoxu/snn-learning/plaining
python main.py
```

依赖：`numpy`, `matplotlib`, `pillow`（仅 conda 环境 `snn-demo` 需要）。

---

> **总结**：这个 demo 的价值不在于物理精度或算法创新，而在于**用最小代价验证了核心假设——反射式神经符号控制闭环可以工作**。
