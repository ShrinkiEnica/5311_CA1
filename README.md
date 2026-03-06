# Autodrive WAM — 车辆动力学世界动作模型

基于 Stanford Vehicle Dynamics Dataset 训练的 **世界动作模型 (World Action Model, WAM)**，
用于学习高动态自动驾驶场景下的车辆动力学状态转移与未来轨迹推演。

---

## 1. 项目概览

| 属性 | 值 |
|------|-----|
| **目标** | 学习 `state_t + action_t -> state_{t+1}`，并支持多步 rollout |
| **数据源** | Stanford Dynamic Design Lab — Thunderhill Raceway Park |
| **车辆** | VW Golf |
| **原始采样率** | 200 Hz (dt = 5 ms) |
| **当前训练频率** | 20 Hz (`downsample=10`, `dt=0.05s`) |
| **训练数据** | 多圈完整 lap（排除 `_5` / `_10` 不完整圈） |
| **核心脚本** | `code/train_wam_v2.py` |
| **基线脚本** | `code/train_wam.py` |

该项目当前的主线不再是单帧位置回归，而是一个**带控制输入的时序世界模型**：

```text
state_t + action_t -> state_{t+1}
```

模型既支持 **teacher forcing 下一步预测**，也支持 **autoregressive dream rollout**，
并可通过动作干预实验观察未来轨迹如何随控制输入变化。

---

## 2. 目录结构

```
autodrive/
├── README.md                           ← 本文件
├── code/
│   ├── fit_baseline_track.py           # 多圈基准赛道拟合
│   ├── train_wam.py                    # v1: 单帧 MLP 基线
│   ├── train_wam_v2.py                 # v2: 当前主线世界模型
│   └── visualize_raw_3d.py             # 原始数据 3D 可视化
├── data/
│   └── raw_data/
│       ├── VehicleDynamicsDataset_Nov2023_2023-11.csv    # Lap_0
│       ├── VehicleDynamicsDataset_Nov2023_2023-11_*.csv  # 其余圈数据
│       ├── baseline_track.csv          # 拟合的赛道中心线
│       └── README.md                   # 数据字段详细说明
├── models/                             # 保存训练好的模型/checkpoint
│   ├── wam_best.pt                     # v1 基线模型
│   ├── wam_v2_best.pt                  # v2 最终保存模型
│   └── wam_v2_best_dream.pt            # 按 dream error 最优模型
├── visualization/
│   ├── raw/                            # 原始数据/赛道相关可视化
│   ├── wam_v1/                         # v1 输出
│   └── wam_v2/                         # v2 输出
├── requirements.txt
└── .venv/
```

---

## 3. 当前主线模型：WAM v2

### 3.1 任务定义

当前主线模型位于 `code/train_wam_v2.py`，其目标是学习：

```
f(state_t, action_t, h_t) -> state_{t+1}
```

其中：

- `state_t` 表示车辆当前动力学状态
- `action_t` 表示当前控制输入
- `h_t` 表示时序隐状态
- `state_{t+1}` 表示下一时刻状态

在推理阶段，模型可以从一个初始状态出发，结合未来动作序列做多步自回归推演：

```text
init_state + action_seq -> future trajectory
```

### 3.2 状态、动作与观测定义

#### 状态 `state`（17 维）

| 类别 | 特征 |
|------|------|
| **位置** | `posE_m`, `posN_m`, `posU_m` |
| **车体速度** | `Vx_mps`, `Vy_mps` |
| **姿态/角速度** | `yawAngle_rad`, `yawRate_radps`, `rollRate_radps`, `pitchRate_radps` |
| **加速度** | `axCG_mps2`, `ayCG_mps2`, `azCG_mps2` |
| **侧滑** | `slipAngle_rad` |
| **轮速** | `wheelspeed_fl`, `wheelspeed_fr`, `wheelspeed_rl`, `wheelspeed_rr` |

#### 动作 `action`（6 维）

| 类别 | 特征 |
|------|------|
| **转向** | `roadWheelAngle_rad` |
| **油门** | `throttleCmd_percent` |
| **制动指令** | `brakeCmd_fl_bar`, `brakeCmd_fr_bar`, `brakeCmd_rl_bar`, `brakeCmd_rr_bar` |

#### 观测 `observation`

模型不会把完整状态原样送入 encoder，而是构造一个更偏动力学的观测：

- 使用部分动态量
- 不直接把位置作为 encoder 输入
- 不直接把速度作为 encoder 输入
- 对 `yaw` 使用 `sin(yaw)` 与 `cos(yaw)` 编码，避免角度周期性问题

这使得模型更倾向于学习动力学演化，而不是记忆绝对位置。

### 3.3 模型架构

`WAM v2` 的结构可以概括为：

```
state -> observation
observation + action -> latent
latent -> temporal model
temporal output -> residual decoder
residual + physics-inspired transition -> next state
```

具体组成如下：

- **Observation Encoder**
  - 将观测编码到 latent 表示

- **Action Encoder**
  - 将控制输入编码到同一 latent 空间

- **Temporal Module**
  - 优先使用 `Mamba`
  - 如果未安装 `mamba_ssm`，则回退到自定义 `SSMBlock`

- **Residual Decoder**
  - 从时序特征解码出状态残差

- **Physics-inspired Transition**
  - 使用显式积分关系更新速度、航向和位置

### 3.4 状态转移设计

模型并不是直接黑盒输出完整的 `state_{t+1}`，而是：

1. 预测部分状态量的残差
2. 用显式动力学关系完成状态更新

核心更新近似为：

```text
vx_next = vx + ax_next * dt
vy_next = vy + ay_next * dt
yaw_next = yaw + yawRate_next * dt
pos_next = pos + R(yaw_next) * [vx_next, vy_next] * dt
```

这样做的目的：

- 引入物理归纳偏置
- 提高长时 rollout 稳定性
- 降低完全黑盒预测的位置漂移

### 3.5 数据预处理

1. **多圈训练**：加载所有完整 lap，排除 `_5` / `_10`
2. **下采样**：从 `200Hz` 下采样到 `20Hz`
3. **按圈切分**：每一圈按时间顺序 `70/30` 切分 train / val
4. **归一化**：仅用训练段统计量做 Z-Score 标准化
5. **不跨 lap 取序列**：滑窗序列只在单圈内部构造

---

## 4. 训练流程

```
加载多圈 CSV
  -> 下采样到 20Hz
  -> 每圈按时间顺序切分 train/val
  -> 训练集统计量归一化
  -> 构造滑窗序列样本
  -> 训练时序世界模型
  -> 一步评估 + dream rollout 评估
  -> 生成轨迹/误差/干预可视化
```

### 4.1 序列样本设置

当前主要配置：

| 配置 | 值 |
|------|-----|
| **下采样** | `10` |
| **训练频率** | `20Hz` |
| **时间步长** | `0.05s` |
| **序列长度** | `100` steps |
| **单段时长** | `5.0s` |
| **Batch Size** | `256` |
| **Epochs** | `200` |

### 4.2 训练策略

当前 v2 使用以下训练机制：

- **Teacher Forcing**
  - 基础的一步状态转移训练

- **Scheduled Sampling**
  - 逐步增加使用模型预测状态作为下一步输入的比例

- **Curriculum Rollout Loss**
  - rollout 长度从短到长增长，逐步提升长时稳定性

- **Trajectory Loss Ramp**
  - 随训练推进，逐步提高轨迹相关损失权重

- **Gradient Clipping**
  - 控制训练不稳定和梯度爆炸

- **Dream Validation**
  - 周期性做长时自回归验证，并用其监控模型表现

### 4.3 优化配置

| 项目 | 值 |
|------|-----|
| **优化器** | `AdamW` |
| **学习率** | `3e-4` |
| **权重衰减** | `1e-5` |
| **学习率策略** | warmup + cosine 衰减 |
| **早停** | 当前关闭 |

### 4.4 模型选择标准

当前 v2 不是按单步 `val loss` 选择最优模型，而是优先参考：

- **dream rollout 的 3D 位置误差**

这比只看一步预测误差更符合世界模型的实际目标。

---

## 5. 评估与可视化

### 5.1 一步评估（Teacher Forcing）

| 指标 | 说明 |
|------|------|
| **3D MAE** | 下一步位置预测平均误差 |
| **3D Max Error** | 下一步最坏情况误差 |
| **Per-dim MAE** | 各位置维度误差 |

### 5.2 多步评估（Dream Rollout）

从验证集初始状态出发，输入未来动作序列，做自回归 rollout，评估：

- **3D MAE**
- **3D Max Error**
- **各位置维度误差**
- **速度误差**
- **不同 horizon（0.5s / 1s / 2s / 5s / 10s）误差**

### 5.3 动作干预实验

当前还提供动作干预可视化：

- 原始动作
- 增大转向
- 减小转向
- 强制制动

用于检查模型是否真正学到了**动作如何改变未来轨迹**。

### 5.4 可视化输出

- **训练曲线**：loss / learning rate
- **Dream 3D 轨迹对比**
- **Dream 俯视图轨迹对比**
- **误差随时间增长曲线**
- **全状态维度对比图**
- **速度预测对比图**
- **动作干预轨迹图**

---

## 6. v1 基线模型

`code/train_wam.py` 保留为项目的 **v1 基线**。

其特点是：

- 输入单帧特征
- 使用 MLP
- 直接回归位置 `posE_m / posN_m / posU_m`
- 适合作为快速 sanity check 和 baseline 对照

它的定位是：

- 验证数据是否可学
- 给 v2 提供一个简单对照实验

但它**不是当前主线世界模型**，因为它不建模显式时序状态转移，也不做 autoregressive rollout。

---

## 7. 基准赛道

从 10 圈完整数据拟合的赛道中心线（排除 Lap_5/Lap_10 不完整圈）：

| 属性 | 值 |
|------|-----|
| **总长** | 2630 m |
| **采样点** | 5272 (间距 0.5 m) |
| **跨圈标准差** | E=0.30m, N=1.47m, U=0.03m |
| **文件** | `data/raw_data/baseline_track.csv` |

基准赛道不是主监督目标，但可作为几何参考，用来检查：

- 预测轨迹是否落在合理范围内
- 多圈轨迹的一致性
- 模型 rollout 是否明显跑偏

---

## 8. 快速开始

```bash
# 激活虚拟环境
source .venv/bin/activate

# 训练 v2 主线模型
python code/train_wam_v2.py

# 训练 v1 基线模型
python code/train_wam.py

# 拟合基准赛道
python code/fit_baseline_track.py
```

训练完成后，可在以下目录查看输出：

- `models/`
- `visualization/wam_v1/`
- `visualization/wam_v2/`
- `visualization/raw/`

---

## 9. 依赖

```
python >= 3.10
numpy
pandas
scipy
plotly
torch
mamba_ssm   # 可选；安装后 v2 将使用 Mamba 时序层
```

说明：

- 如果未安装 `mamba_ssm`，`train_wam_v2.py` 会自动回退到自定义 `SSMBlock`
- 这意味着项目在无 Mamba 环境下也可以运行，只是时序模块实现不同

---

## 10. 当前项目定义

可以把本项目理解为：

> 一个基于高频车辆动力学数据的、带控制输入的时序世界模型。
> 当前主模型通过 observation/action 编码、时序状态空间模块和带物理先验的 transition，
> 学习车辆状态转移，并以 autoregressive dream rollout 作为核心评估方式。
