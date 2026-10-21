# TorchYAND

[English](README.md) | [简体中文](README.zh-CN.md)

**基于 PyTorch 张量化的高阶矩投资组合优化。**

TorchYAND 是 Yau Affine-Normal Descent（YAND）在均值–方差–偏度–峰度（MVSK）投资组合优化中的 PyTorch 实现。项目将矩阵自由导数算子、单纯形几何、批量迭代线性代数、多项式线搜索与 KKT 诊断结合在支持 CPU/CUDA 的数值计算库中。

![MIT](https://img.shields.io/badge/license-MIT-0f766e) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c) ![Version](https://img.shields.io/badge/release-0.1.0-0f766e)

**导航：** [数学设计](TorchYAND_MVSK_Design_Spec.md) · [性能脚本](benchmarks/) · [T4 原始实验](reports/sae-issue21/summary.md) · [图表证据与来源](docs/assets/README.md)

## 核心特色

- **高阶矩，无须构造高阶矩张量：** 利用中心化收益投影计算目标、梯度、Hessian-vector product（HVP）与三阶方向导数。
- **矩阵自由约束优化：** 仿射法向方向、单纯形活动面投影、投影 Hessian 算子及批量 PCG。
- **多项式数值算法：** 三次方程实根求解与可行四次多项式线搜索。
- **原生 PyTorch：** 支持 float32/float64、CPU/CUDA 及投资组合前导批量维度。
- **可检查的收敛诊断：** 明确的求解器状态、KKT 残差与可复现实验入口。

## 数学模型

设收益矩阵 `R ∈ ℝ^(T×n)`、平均收益 `μ`、中心化矩阵 `A = R − 1μᵀ`、投资组合 `x` 及 `z = Ax`。实现的样本高阶矩目标为

```text
minimize  f(x) = -c1 * μᵀx + c2 * mean(z²) - c3 * mean(z³) + c4 * mean(z⁴)
subject to x >= 0,  1ᵀx = 1;   c1,c2,c3,c4 >= 0
```

这里使用的是**中心样本矩**，不是标准化偏度或超额峰度。公共 oracle 使用共享系数 `[c1,c2,c3,c4]`，通过解析导数计算，避免显式构造协方差、协偏度、协峰度及稠密 Hessian。令 `w = 2c2 − 6c3 z + 12c4 z²`，则

```text
∇f(x) = -c1 μ + Aᵀ(2c2 z − 3c3 z² + 4c4 z³)/T
∇²f(x)v = Aᵀ(w ⊙ (Av))/T
```

[数学设计规范](TorchYAND_MVSK_Design_Spec.md) 进一步介绍几何、精确/随机 logdet 修正及多项式推导。

## 算法架构

![YAND 数值求解流程](docs/assets/yand-architecture.svg)

正式 v0.1 求解器采用 **eager 模式**、自适应 PCG 与活动集处理。整体流程将 MVSK oracle、仿射法向方向、可行四次线搜索与完整单纯形 KKT 诊断组合起来。图示是组件概览，不代表逐函数调用图。

![矩阵自由 MVSK 张量运算](docs/assets/matrix-free-operations.svg)

矩阵自由 oracle 利用 `Z = X @ A.T` 的投影表示复用高阶矩及导数计算；支持前导批量维度，资产维度固定。

## 安装与快速开始

在仓库检出目录安装（项目声明 Python >=3.10、PyTorch >=2.0）：

```bash
pip install -e .
```

```python
import torch
from torchyand import prepare_mvsk_data, MVSKOracle
from torchyand.solver import (
    YANDOptions,
    solve_yand,
    SolverStatus,
)

rng = torch.Generator().manual_seed(123)

returns = 0.01 * torch.randn(
    256,
    16,
    dtype=torch.float64,
    generator=rng,
)

data = prepare_mvsk_data(returns)

oracle = MVSKOracle(
    data,
    coefficients=[1.0, 1.0, 0.2, 0.1],
)

x0 = torch.full(
    (16,),
    1 / 16,
    dtype=returns.dtype,
)

result = solve_yand(
    oracle,
    x0,
    options=YANDOptions(
        logdet_mode="stochastic",
    ),
    seed=123,
)

print(
    SolverStatus(result.status.item()).name,
    result.converged.item(),
)

print(result.x)
print(result.value.item())
print(result.kkt_residual.item())
```

`prepare_mvsk_data` 和 `MVSKOracle` 从 `torchyand` 导入；`solve_yand`、`YANDOptions`、`YANDMVSK`、`YANDResult` 和 `SolverStatus` 位于 `torchyand.solver`。`result.converged`、`result.status` 与 `result.kkt_residual` 用于判断终止状态。对于非凸 MVSK，KKT 收敛代表驻点条件，并不等于全局最优。

## 性能：CPU 与真实 Tesla T4

![同一主机 CPU 与 T4 eager MVSK oracle 测试](docs/assets/mvsk-oracle-benchmark.svg)

[真实 Colab 科学实验](reports/sae-issue21/summary.md) 测量公共 **eager MVSK `value_grad` oracle**：`T=2048`、`n=256`、batch 256、float32。每组 2 次预热、7 次同步计时：

| 测量范围 | 中位数 | 相对 CPU |
|---|---:|---:|
| CPU 驻留计算，2 线程 | 12.020 ms | 1.00× |
| T4 驻留计算 | 1.102 ms | 10.91× |
| T4 包含传输、oracle 构造及结果返回 | 2.206 ms | 5.45× |

CPU/T4 使用同一 Colab 主机及对应数据。这是 **oracle 耗时，而非完整求解器耗时**；不包括数据生成与中心化。[原始报告](reports/sae-issue21/summary.md) 保留数值一致性、环境、计时与精确源码版本。报告发布时科学验收仍待完成。

另一次**实验性编译固定 PCG** 重复求解实验记录 T4 约 **22.9 s → 9.6 s（2.38×）**，首次编译约 11 分钟，预计约重复 51 次求解达到收支平衡。该路径没有取代正式 eager 自适应求解器。历史数据见[固定版本英文 README](https://github.com/JohnnyZPiggy/TorchYAND/blob/1ee2a100db61bc3be62ced9cc7923f261b512b35/README.md) 与[实验脚本](benchmarks/bench_yand_compiled_pcg.py)。

## 科研应用

TorchYAND 的数值组件也可用于相关研究：

| 方向 | 对应数值组件 |
|---|---|
| 高阶矩资产配置 | 单纯形约束下的均值、方差、三阶和四阶中心矩 |
| 批量约束优化 | 活动集掩码、投影算子、KKT 监控 |
| 可微模型分析 | 解析梯度、HVP 与三阶方向导数；训练集成需另行实现 |
| GPU 数值计算研究 | 张量化多项式求解、矩阵自由 PCG、编译对照实验 |

这些是潜在研究方向，已交付求解器的主要目标仍是 MVSK 投资组合优化。

## 求解器、验证与支持范围

| 组件 | 状态 |
|---|---|
| MVSK oracle、导数、几何、多项式求解器 | 已实现 |
| eager YAND、自适应 PCG、单纯形约束及 KKT 状态 | 已实现 |
| 编译固定 PCG 性能实验 | 实验路径 |
| 历史 TPU/XLA 工具 | 保留研究记录，非正式生产后端 |

历史 Colab T4 环境（PyTorch 2.11.0+cu130）记录 **1,269 项测试通过**。较新的 [SAE 科学实验](reports/sae-issue21/summary.md) 独立记录 4 项 CPU 测试，以及有界 CPU/CUDA oracle 与二次规划检查；两者不是同一轮测试。

支持 float32/float64；投资组合形状 `[..., n]`，共享四个目标系数，输入设备和精度与 oracle 一致。求解器状态包括 `CONVERGED_KKT`、`STALLED`、`MAX_ITER`、`DIRECTION_FAILED`、`LINE_SEARCH_FAILED`、`INVALID_STATE`。严格收敛研究建议使用 float64。

```bash
pip install -e ".[test]"
python -m pytest tests/ -q
```

性能复现可查阅[实验脚本](benchmarks/)及各自的测量范围。编译实验为可选路径，需适配的编译工具链。

## 项目结构

```text
torchyand/
  data/         oracle/       geometry/       operators/
  linalg/       logdet/       polynomial/     constraints/
  line_models/  line_search/  diagnostics/    solver/
tests/          benchmarks/   experiments/    reports/
```

## 许可证

MIT，详见 [LICENSE](LICENSE)。研究报告及历史实验记录与公共 API 分开保留。
