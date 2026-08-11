# v2 迭代记录：问题修复与实验实现

> 对应师兄反馈的 4 个问题 + 4 个实验建议。记录每个改动的具体实现和代码位置。

---

## 问题 1: No Reflex 空白图 + x 轴 bug

### 根因

[environment.py:252](environment.py#L252) — `run_episode` 中 `if reflex_mode != 'none'` 直接跳过了事件检测和概念激活，导致 No Reflex 模式下三张子图全空。

### 修复

将事件/概念检测从条件分支中提出来，**始终执行**，只对 reflex_action 做 gate。

**代码位置**: [environment.py:252-272](environment.py#L252-L272)

```python
# v1 (bug): event detection inside reflex_mode check
if reflex_mode != 'none' and event_detector is not None:
    events = event_detector.detect(...)   # ← NoReflex 时完全跳过

# v2 (fix): always detect, only gate reflex_action
if event_detector is not None:
    events = event_detector.detect(...)   # ← 始终执行
    concepts = concept_layer.activate(...)
    if reflex_mode in ('rule', 'snn_rule', ...):
        reflex_action = rule_layer.decide(...)
```

**x 轴修复**: [visualization.py:118](visualization.py#L118) — 所有子图加 `ax.set_xlim(0, 5)`。

### 效果

`output/v1/baseline_comparison.png` — No Reflex 列现在显示完整的事件和概念曲线，x 轴统一 0–5s。

---

## 问题 2: Rule 与 SNN 无差异化 → 延迟实验

### 根因

v1 中 Rule 和 SNN 都立即响应，延迟相同 (30ms)，无法体现 SNN 作为快速旁路的价值。

### 实现：Rule 延迟机制

**代码位置**: [environment.py:266-280](environment.py#L266-L280)

```python
# 延迟队列：将 rule 决策推迟 delay_ms 后执行
pending_rule_action: deque = deque()
delay_steps = int(rule_delay_ms / 1000.0 / cfg.dt)

# 规则决策入队
if immediate_action is not None and reflex_enabled:
    if delay_steps > 0:
        pending_rule_action.append((t + rule_delay_ms / 1000.0, immediate_action))
    else:
        reflex_action = immediate_action

# 到期出队
while pending_rule_action and pending_rule_action[0][0] <= t:
    _, delayed_action = pending_rule_action.popleft()
    if reflex_action is None:
        reflex_action = delayed_action
```

### 实验函数

**代码位置**: [environment.py:316-356](environment.py#L316-L356) — `run_delay_experiment()`

4 组对比：
| 组 | reflex_mode | rule_delay_ms | SNN |
|----|-------------|---------------|-----|
| no_reflex | 'none' | — | — |
| rule_delayed | 'rule' | 80 | — |
| snn_only | 'snn' | — | ✓ |
| snn_plus_delayed_rule | 'snn_rule' | 80 | ✓ |

### 效果

`output/v2/02_delay_experiment.png`

```
rule_delayed:  110ms latency → ✗ 物体掉落
snn_only:       20ms latency → ✓ 恢复成功
```

**这就是核心故事**：高层规则推理延迟导致失败，SNN 作为快速旁路保住物体。

---

## 问题 3: tighten 和 regrasp 同时激活 → 动作区分

### 根因

v1 SNN 只有 2 个输出神经元 `[tighten, regrasp]`，没有调谐偏置，任何异常事件都同时激活两者。

### 实现：3 输出 + 调谐偏置

**代码位置**: [snn_reflex.py:73-82](snn_reflex.py#L73-L82) — 隐藏层神经元调谐偏置

```python
# 前 5 个隐藏神经元：偏好 slip+contact 模式
for i in range(5):
    self.hidden.neurons[i].tuning_bias = np.array([0.2, 0.2, -0.5, 0.3, 0.3])
# 后 5 个：偏好 contact_loss+falling 模式
for i in range(5, 10):
    self.hidden.neurons[i].tuning_bias = np.array([-0.3, 0.3, 0.8, -0.2, -0.2])
```

偏置通过点积增强匹配模式的输入电流：

```python
if self.neurons[i].tuning_bias is not None:
    pattern_match = np.dot(self.neurons[i].tuning_bias, inputs)
    currents[i] *= (1.0 + 0.5 * pattern_match)
```

**3 输出神经元**: [snn_reflex.py:91-97](snn_reflex.py#L91-L97)

```
[0] tighten  — tau=8ms, gain=8
[1] regrasp  — tau=8ms, gain=8
[2] force_delta — tau=8ms, gain=10 (连续力调节)
```

**决策逻辑**: [snn_reflex.py:121-143](snn_reflex.py#L121-L143)

```python
if regrasp_conf > 0.3 and contact_loss > 0.3:
    action = 7  # regrasp — 接触丢失
elif slip_risk > 0.5 and vel_anomaly > 0.7:
    action = 6  # 紧急 binary tighten (18N)
elif tighten_conf > 0.3 and slip_risk > 0.2:
    action = 8  # 自适应 tighten (比例力)
```

### 效果

`output/v2/05_action_differentiation.png`

```
slip + contact preserved → tighten_adaptive   ✓ CORRECT
slip + contact lost      → tighten_adaptive   ✗ (单次反射限制，
                                                 滑移事件先于接触丢失事件)
```

> **已知局限**：严重滑移时 `urgent_slip_binary` 路径先触发（action=6），消耗掉唯一的 reflex 机会，导致后续 contact_loss 无法触发 regrasp。后续可加入 reflex cooldown + re-trigger 机制。

---

## 问题 4: 暴力 18N 夹紧 → 自适应力度

### 根因

v1 只有 binary tighten（直接 18N），导师会问"这不就是使劲夹吗，有什么智能？"

### 实现：三级力度响应

**代码位置**: [snn_reflex.py:136-143](snn_reflex.py#L136-L143) + [environment.py:128-138](environment.py#L128-L138)

```
紧急 (slip_risk>0.5, vel>0.7) → action=6, binary 18N 急停
中等 (slip_risk>0.2)          → action=8, adaptive +5~13N 比例调节
轻微 (tighten_conf>0.3)        → action=8, adaptive +3N 最小干预
```

**连续力输出**: [snn_reflex.py:109](snn_reflex.py#L109)

```python
force_delta = clip(output_spikes[2] * 25.0, 0.0, 13.0)
```

**环境侧自适应执行**: [environment.py:131-138](environment.py#L131-L138)

```python
elif reflex_action == 8:  # adaptive tighten
    self.reflex_type = 'tighten_adaptive'
    self.gripper_force = self.gripper_force + grip_force_delta
    self.gripper_force = clip(self.gripper_force, 5.0, 18.0)
```

### 力效率指标

**代码位置**: [environment.py:95](environment.py#L95)

```python
self.force_integral += self.gripper_force * self.cfg.dt  # N·s, 能量代理
```

`output/v2/force_efficiency.png` 对比各方法的 Max Force 和 Force Integral，体现"用更小的力实现同样的恢复"。

---

## 实验 B: 多滑移强度鲁棒性

**代码位置**: [environment.py:358-387](environment.py#L358-L387) — `run_slip_magnitude_sweep()`

遍历 `slip_magnitude = [4, 6, 8, 10, 12]`，每个强度跑三组 (NoReflex, Rule, SNN+Rule)。

`output/v2/03_slip_sweep.png`

```
mag=4:  全部恢复 (轻扰动，正常抓取力就够)
mag=6-10: Rule+SNN 恢复，NoReflex 掉落
mag=12:  SNN 也开始失败 (极端扰动超出反射能力)
```

---

## 实验 C: 噪声鲁棒性

**代码位置**: [environment.py:389-419](environment.py#L389-L419) — `run_noise_experiment()`

**噪声注入**: [environment.py:87-89](environment.py#L87-L89)

```python
def _add_noise(self, state):
    if self.cfg.noise_std > 0:
        return state + np.random.randn(*state.shape) * self.cfg.noise_std
    return state
```

噪声水平 `[0, 0.01, 0.02, 0.05]`，对 state 向量加高斯噪声，对比 Rule 和 SNN 的误触发率和恢复率。

`output/v2/04_noise_robustness.png`

当前结果：两种方法在噪声下表现接近，因为 reflex 有 `was_grasped + gripper_z > 0.08` 的门控保护。要进一步体现 SNN 优势需要去掉门控或加大噪声。

---

## 实验 D: 动作区分 tighten vs regrasp

**代码位置**: [environment.py:421-461](environment.py#L421-L461) — `run_action_differentiation_experiment()`

两个场景：
1. `slip_magnitude=8` (接触保留) → 预期 tighten
2. `slip_magnitude=14` (接触丢失) → 预期 regrasp

结果：场景 1 ✓，场景 2 ✗（原因见问题 3）。

---

## 关键代码变更清单

| 文件 | 变更 |
|------|------|
| [environment.py](environment.py) | +`rule_delay_ms`, +`noise_std`, +`adaptive_grip`, +`grip_force_delta`, +`force_integral`, +4 个新实验函数, 修复 NoReflex 事件检测 |
| [snn_reflex.py](snn_reflex.py) | 2→3 输出神经元, +`tuning_bias`, +`force_delta`, +三级决策逻辑 |
| [visualization.py](visualization.py) | 重写全部 7 张图, +`xlim(0,5)` 修复, +`plot_delay_experiment`, +`plot_slip_sweep`, +`plot_noise_experiment`, +`plot_action_differentiation`, +`plot_force_efficiency` |
| [main.py](main.py) | 运行全部 5 个实验, 输出到 `output/v2/` |
| [main_v1.py](main_v1.py) | v1 独立入口, 输出到 `output/v1/` |

---

## 后续迭代方向

1. **reflex re-trigger**: 加 cooldown 机制，允许 contact_loss 后二次触发 regrasp
2. **STDP 学习**: 替换随机权重，用 STDP 从滑移经验中学习调谐偏置
3. **真机延迟测量**: 在 PyBullet/RLBench 中测量真实控制回路延迟
4. **多级反射**: small→tighten, medium→regrasp, severe→stop+withdraw
