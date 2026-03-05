# Autodrive WAM - Vehicle Dynamics Dataset

基于 Stanford Dynamic Design Lab 的 **Vehicle Dynamics Dataset for Highly Dynamic Automated Driving** 构建的自动驾驶世界模型（WAM）项目。

- **数据来源**: [Stanford Digital Repository](https://purl.stanford.edu/hh613qz0317) / [GitHub 5311_CA1](https://github.com/ShrinkiEnica/5311_CA1)
- **DOI**: https://doi.org/10.25740/hh613qz0317

---

## 数据集概览

| 属性 | 值 |
|------|------|
| **车辆** | VW Golf |
| **赛道** | Thunderhill Raceway Park West Track |
| **文件数** | 12 个 CSV |
| **总行数** | 268,582 |
| **总大小** | ~204 MB |
| **采样率** | 200 Hz (dt = 5ms) |
| **单圈时长** | ~90-130 s |
| **字段数** | 58 |
| **缺失值** | 无 (所有字段完整) |

---

## 文件列表

| 文件名 | 行数 | 大小 |
|--------|------|------|
| `VehicleDynamicsDataset_Nov2023_2023-11.csv` | 22,304 | 16.9 MB |
| `VehicleDynamicsDataset_Nov2023_2023-11_1.csv` | 21,446 | 16.2 MB |
| `VehicleDynamicsDataset_Nov2023_2023-11_2.csv` | 21,286 | 15.9 MB |
| `VehicleDynamicsDataset_Nov2023_2023-11_3.csv` | 22,065 | 16.6 MB |
| `VehicleDynamicsDataset_Nov2023_2023-11_4.csv` | 22,241 | 17.0 MB |
| `VehicleDynamicsDataset_Nov2023_2023-11_5.csv` | 12,655 | 9.4 MB |
| `VehicleDynamicsDataset_Nov2023_2023-11_6.csv` | 27,004 | 20.6 MB |
| `VehicleDynamicsDataset_Nov2023_2023-11_7.csv` | 25,005 | 19.2 MB |
| `VehicleDynamicsDataset_Nov2023_2023-11_8.csv` | 22,205 | 17.0 MB |
| `VehicleDynamicsDataset_Nov2023_2023-11_9.csv` | 28,904 | 22.1 MB |
| `VehicleDynamicsDataset_Nov2023_2023-11_10.csv` | 19,961 | 15.4 MB |
| `VehicleDynamicsDataset_Nov2023_2023-11_11.csv` | 23,506 | 17.8 MB |

每个文件前两行为注释（`# Vehicle: VW Golf` / `# Track: ...`），读取时需 `comment='#'`。

---

## 字段说明 (58 列)

### 时间与位置

| 字段 | 单位 | 说明 | 范围 |
|------|------|------|------|
| `t_s` | s | 时间戳 | 0 ~ 111.5 |
| `posE_m` | m | 东向位置 (GPS) | -993 ~ -608 |
| `posN_m` | m | 北向位置 (GPS) | -608 ~ 275 |
| `posU_m` | m | 高程位置 (GPS) | -5.8 ~ 18.0 |
| `posRMSerror_m` | m | GPS 定位误差 (RMS) | 0.006 ~ 0.009 |
| `s_m` | m | 沿赛道弧长 | 0 ~ 2683 |

### 速度

| 字段 | 单位 | 说明 | 范围 |
|------|------|------|------|
| `Ve_mps` | m/s | 东向速度 (GPS 坐标系) | -26.1 ~ 27.5 |
| `Vn_mps` | m/s | 北向速度 (GPS 坐标系) | -30.3 ~ 29.1 |
| `Vu_mps` | m/s | 垂直速度 (GPS 坐标系) | -2.6 ~ 1.9 |
| `Vx_mps` | m/s | 纵向速度 (车体坐标系) | 0 ~ 30.3 |
| `Vy_mps` | m/s | 横向速度 (车体坐标系) | -1.6 ~ 1.3 |
| `Vz_mps` | m/s | 垂直速度 (车体坐标系) | -2.6 ~ 1.9 |

### 姿态角

| 字段 | 单位 | 说明 | 范围 |
|------|------|------|------|
| `rollAngle_rad` | rad | 横滚角 | -0.085 ~ 0.052 |
| `pitchAngle_rad` | rad | 俯仰角 | -0.085 ~ 0.078 |
| `yawAngle_rad` | rad | 偏航角 | -6.28 ~ 0 |

### 角速度

| 字段 | 单位 | 说明 | 范围 |
|------|------|------|------|
| `rollRate_radps` | rad/s | 横滚角速度 | -0.45 ~ 0.17 |
| `pitchRate_radps` | rad/s | 俯仰角速度 | -0.30 ~ 0.16 |
| `yawRate_radps` | rad/s | 偏航角速度 | -0.73 ~ 0.80 |

### 加速度

| 字段 | 单位 | 说明 | 范围 |
|------|------|------|------|
| `axCG_mps2` | m/s2 | 纵向加速度 (重心) | -5.5 ~ 5.1 |
| `ayCG_mps2` | m/s2 | 横向加速度 (重心) | -21.0 ~ 22.2 |
| `azCG_mps2` | m/s2 | 垂直加速度 (重心) | -7.2 ~ 26.9 |

### 转向与侧滑

| 字段 | 单位 | 说明 | 范围 |
|------|------|------|------|
| `roadWheelAngle_rad` | rad | 前轮转角 | -0.18 ~ 0.48 |
| `slipAngle_rad` | rad | 车辆侧滑角 | -0.06 ~ 0.12 |

### 制动系统

| 字段 | 单位 | 说明 |
|------|------|------|
| `brake_fl/fr/rl/rr_bar` | bar | 四轮实际制动压力 |
| `brakeCmd_fl/fr/rl/rr_bar` | bar | 四轮制动指令压力 |
| `brake_fl/fr/rl/rr_N` | N | 四轮制动力 |
| `ABSactive_bool` | bool | ABS 是否激活 |

### 动力系统

| 字段 | 单位 | 说明 | 范围 |
|------|------|------|------|
| `engineTorque_Nm` | Nm | 发动机扭矩 | -72 ~ 376 |
| `engine_rpm` | rpm | 发动机转速 | 1051 ~ 6374 |
| `throttleCmd_percent` | % | 油门指令 | 0 ~ 100 |
| `clutchEngage_int` | int | 离合器状态 | 0 ~ 4 |
| `GearRatio` | - | 当前传动比 | 7.16 ~ 35.07 |
| `LSDengage_percent` | % | 限滑差速器接合度 | 0 ~ 46.8 |

### 轮速

| 字段 | 单位 | 说明 |
|------|------|------|
| `wheelspeed_fl/fr/rl/rr` | rad/s | 四轮转速 |

### 轮胎温度

| 字段 | 单位 | 说明 |
|------|------|------|
| `tireTemp_fl/fr/rl/rr_degC` | degC | 四轮轮胎温度 (25~60 degC) |

### 悬架

| 字段 | 单位 | 说明 |
|------|------|------|
| `suspFLpos_m` / `suspFRpos_m` / `suspRLpos_m` | m | 前左/前右/后左悬架位移 |

### 自动驾驶状态

| 字段 | 类型 | 说明 |
|------|------|------|
| `auto_steer_bool` | bool | 自动转向是否启用 |
| `auto_throttle_bool` | bool | 自动油门是否启用 |

### 赛道几何

| 字段 | 单位 | 说明 |
|------|------|------|
| `grade_rad` | rad | 赛道纵坡 |
| `bank_rad` | rad | 赛道横坡 |
| `vertical_curvature_radpm` | rad/m | 赛道垂直曲率 |

---

## 数据特点

- **高频采样 (200Hz)**: 适合学习连续动力学状态转移
- **极限工况**: 横向加速度达 +/-22 m/s2 (>2g)，包含漂移、急弯等高动态场景
- **完整动力链**: 从油门/刹车指令 -> 发动机/制动器 -> 轮速/轮胎 -> 车辆运动，全链路闭环
- **自动驾驶标注**: `auto_steer_bool` / `auto_throttle_bool` 标记人工/自动驾驶段
- **赛道几何信息**: 坡度、横坡、曲率，可用于地形感知

---

## WAM 适用性

该数据集非常适合训练车辆动力学世界模型 (WAM):

- **状态 (State)**: `[posE, posN, Vx, Vy, yawAngle, yawRate, ...]`
- **动作 (Action)**: `[roadWheelAngle, throttleCmd, brakeCmd, ...]`
- **转移函数**: `state(t+1) = f(state(t), action(t))`
- **训练数据量**: 268K 帧 @ 200Hz，远超 NBA 弹道数据集

---

## 快速加载

```python
import pandas as pd

df = pd.read_csv("VehicleDynamicsDataset_Nov2023_2023-11.csv", comment="#")
print(df.shape)  # (22304, 58)
```
