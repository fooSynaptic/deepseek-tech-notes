# Muon 优化器

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §3.7](../01-总览/01-版本演进总览.md#37-deepseek-v4) · [V4 梗概](03-V4.md) · [CSA/HCA](05-CSA-HCA混合压缩注意力.md) · [mHC](04-mHC流形约束超连接.md) · [Hash MoE + FP4](06-Hash-MoE-FP4.md)
> **论文**：[DeepSeek-V4 arXiv:2606.19348](https://arxiv.org/abs/2606.19348) · **Algorithm 1** · **§2.4 Muon Optimizer** · **§3.4.1 Efficient Implementation of Muon**

## 核心结论摘要

- **Muon** 用矩阵正交化更新替代大部分 AdamW，加速收敛。
- V4 训练侧优化器替换，与架构改动同期引入。
- 面向大矩阵参数（attention/FFN 投影）的高效二阶风格更新。
- 与 MoE/注意力结构改动并列，属训练 recipe 翻新。

---

## 一句话

**Muon**（Momentum Orthogonalized by Newton–Schulz）在 V4 中**替换大部分矩阵参数的 AdamW**：对动量矩阵做 **Newton–Schulz 正交化**得到更新方向 $UV^\top$（极因子），各奇异方向步长归一，从而**更快收敛、训练更稳**；嵌入、输出头、RMSNorm、mHC 门控等仍用 **AdamW**。

---

## V4 中的位置

| 方向 | 文档 |
|------|------|
| **本专文** | [§1 一步在做什么](#muon-one-step) · [§2 Hybrid Newton–Schulz](#hybrid-newton-schulz) · [§3 参数分组](#adamw-vs-muon-split) · [§4 训练 infra](#muon-zero-buckets) |
| **版本总览** | [DeepSeek-V4](03-V4.md)（CSA/HCA、mHC、MoE、训练要点） |
| **演进总览** | [§3.7 DeepSeek-V4](../01-总览/01-版本演进总览.md#37-deepseek-v4) |
| **三轴图** | [架构-train 轴上的 Muon](../01-总览/01-版本演进总览.md)（100% 训练侧） |

---

## 1. Muon 一步在做什么

<a id="muon-one-step"></a>

V3 全系 **AdamW**：对每个参数做元素级二阶矩估计。V4 对 **二维权重矩阵** $W\in\mathbb{R}^{n\times m}$ 改用 **Algorithm 1**：

$$
G_t=\nabla_W\mathcal{L}_t,\quad
M_t=\mu M_{t-1}+G_t,\quad
O'_t=\mathrm{HybridNewtonSchulz}(\mu M_t+G_t),\quad
O_t=O'_t\cdot\sqrt{\max(n,m)}\cdot\gamma,\quad
W_t=W_{t-1}(1-\eta\lambda)-\eta O_t
$$

其中 **Nesterov 技巧**体现在对 $\mu M_t+G_t$ 做正交化（而非仅 $M_t$）；$\gamma=0.18$ 用于 **复用 AdamW 学习率超参**（与 Kimi Muon 做法一致）。

<img src="figures/v4/muon-optimizer.svg" alt="V4 参数 AdamW/Muon 分组；Algorithm 1 流水线；Hybrid Newton-Schulz 两阶段系数" width="920"/>

[图示详情](figures/v4/muon-optimizer.svg)

### 1.1 与 AdamW 的几何差异

设梯度（或动量）矩阵 $M=U\Sigma V^\top$（SVD）：

| | **AdamW** | **Muon** |
|---|-----------|----------|
| 更新形状 | 保留 $M$ 的各向异性（$\Sigma$ 仍在） | 近似 **$UV^\top$**，奇异值全压到 1 |
| 步长语义 | 逐元素自适应 | **各方向等幅**的矩阵步 |
| 适用参数 | 向量、标量、矩阵均可 | **矩阵参数**为主 |

直观上：AdamW 沿「梯度自然拉伸」的方向走；Muon 先 **去掉 $\Sigma$ 的缩放**，只保留方向结构，再在正交方向上统一步长——论文报告在 **万亿级 MoE** 上收敛更快、对学习率 schedule 更不敏感。

### 1.2 V4 为何不用 QK-Clip

Kimi Muon 曾配合 **QK-Clip** 抑制 attention logit 爆炸。V4 在 **query 与 KV entry 上直接做 RMSNorm**（配合 CSA/HCA 架构），logit 已受控，故 **不再使用 QK-Clip**（论文 §2.4）。

---

## 2. Hybrid Newton–Schulz 正交化

<a id="hybrid-newton-schulz"></a>

显式 SVD 每步太贵；Muon 用 **Newton–Schulz 迭代**近似极因子 $UV^\top$。先将 $M$ 按 Frobenius 范数归一化 $M_0=M/\|M\|_F$，再重复：

$$
M_k=aM_{k-1}+b(M_{k-1}M_{k-1}^\top)M_{k-1}+c(M_{k-1}M_{k-1}^\top)^2M_{k-1}
$$

**V4 相对 Kimi Muon 的改动**：**混合两阶段、共 10 步**——

| 阶段 | 步数 | 系数 $(a,b,c)$ | 作用 |
|------|------|------------------|------|
| **快速逼近** | 1–8 | $(3.4445,\,-4.7750,\,2.0315)$ | 奇异值快速靠近 1 |
| **精修稳定** | 9–10 | $(2,\,-1.5,\,0.5)$ | 将奇异值稳定在 1 |

下面代码与论文 Algorithm 1 / 式 (28) 对齐，便于对照 **HybridNewtonSchulz** 与单步 Muon 更新（教学用 NumPy，非 V4 生产实现）：

```python
import numpy as np

# V4 Hybrid Newton-Schulz: 8 fast + 2 stable (paper §2.4)
NS_STAGES = [
    (8, (3.4445, -4.7750, 2.0315)),
    (2, (2.0, -1.5, 0.5)),
]


def newton_schulz_step(m: np.ndarray, abc: tuple[float, float, float]) -> np.ndarray:
    a, b, c = abc
    mm = m @ m.T
    return a * m + b * (mm @ m) + c * (mm @ mm @ m)


def hybrid_newton_schulz(m: np.ndarray) -> np.ndarray:
    """Approximate polar factor UV^T; input M is n×m."""
    m = m.astype(np.float64)
    norm = np.linalg.norm(m, ord="fro")
    if norm == 0:
        return m
    m = m / norm
    for n_steps, coeffs in NS_STAGES:
        for _ in range(n_steps):
            m = newton_schulz_step(m, coeffs)
    return m


def muon_step(
    w: np.ndarray,
    grad: np.ndarray,
    momentum: np.ndarray,
    *,
    lr: float = 2.7e-4,
    mu: float = 0.95,
    weight_decay: float = 0.1,
    gamma: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    """One Muon step for matrix W (Algorithm 1)."""
    momentum = mu * momentum + grad
    nesterov = mu * momentum + grad
    ortho = hybrid_newton_schulz(nesterov)
    scale = np.sqrt(max(w.shape)) * gamma
    update = ortho * scale
    w = w * (1 - lr * weight_decay) - lr * update
    return w, momentum
```

**读法**：`hybrid_newton_schulz` 输出近似正交矩阵；`sqrt(max(n,m)) * gamma` 把更新 RMS 缩放到与 AdamW 超参兼容的量级；最后 **decoupled weight decay** 与 AdamW 形式相同。

---

## 3. 哪些参数用 Muon，哪些仍用 AdamW

<a id="adamw-vs-muon-split"></a>

| 优化器 | 模块（论文 Basic Configurations） |
|--------|-----------------------------------|
| **AdamW** | embedding、prediction head、**所有 RMSNorm 权重**、mHC **静态 bias / 门控因子** |
| **Muon** | **其余全部**（Attention 投影、FFN、MoE expert 等矩阵权重） |

**动机**：正交化是 **矩阵级** 操作；embedding / LayerNorm 类 **向量或标量** 参数不适合 Muon，继续 AdamW 更自然。

V4-Flash / V4-Pro **共用** Muon 超参：momentum **0.95**、weight decay **0.1**、$\gamma=0.18$；AdamW 侧 $\beta_1=0.9$、$\beta_2=0.95$、$\varepsilon=10^{-20}$、weight decay **0.1**（§4 训练设置）。

---

## 4. 训练 infra：Muon 与 ZeRO

<a id="muon-zero-buckets"></a>

Muon 需要 **完整梯度矩阵** 才能做 NS 迭代，与传统 **ZeRO 按元素切分 AdamW 状态** 冲突。V4 训练框架（§3.4.1）要点：

| 问题 | 做法 |
|------|------|
| ZeRO vs 整矩阵更新 | **混合 ZeRO bucket 分配**：限制 ZeRO 并行度；用 **knapsack** 把参数矩阵分到各 rank，负载均衡 |
| bucket 对齐 | 各 rank bucket **pad 到同大小** 以便 reduce-scatter；padding 开销 **< 10%** |
| 超大 DP | 超出 ZeRO 上限的 DP 组内 **冗余计算** Muon 更新，换更少 bucket 内存 |
| NS 吞吐 | 同 shape 连续参数 **合并 batch** 跑 NS |
| 精度 | NS 在 **BF16** matmul 下仍稳定；MoE 梯度跨 DP **随机舍入到 BF16** 同步，通信量减半 |
| 累加 | reduce-scatter 改为 **all-to-all + 本地 FP32 求和**，避免低精度树形累加误差 |

Muon 与 CSA/HCA、mHC、Hash MoE **同期打包**进 V4，**难以单独 ablation** 优化器 vs 架构（见 [V4 定位](03-V4.md#定位)）。

---

## 5. 演进链小结

| 边 | 关系 |
|----|------|
| V3 AdamW → V4 Muon | **训练侧**优化器替换；**推理权重格式不变**（读者侧无「Muon kernel」） |
| Muon ⊥ CSA/HCA | 优化器 vs Attention 算子 **正交** |
| Muon ⊥ Hash MoE / FP4 | 优化器 vs MoE 路由/量化 **正交** |
| Muon ↔ mHC | mHC **门控**仍 AdamW；主体矩阵 Muon |

---

## 6. 上下游

| 方向 | 文档 |
|------|------|
| 版本梗概 | [DeepSeek-V4](03-V4.md) |
| 同代 Attention | [CSA / HCA](05-CSA-HCA混合压缩注意力.md) |
| 同代残差 | [mHC](04-mHC流形约束超连接.md) |
| 同代 MoE | [Hash MoE + FP4](06-Hash-MoE-FP4.md) |
| 演进总览 | [§3.7 V4](../01-总览/01-版本演进总览.md#37-deepseek-v4) |

## 参考

- 论文：[arXiv:2606.19348](https://arxiv.org/abs/2606.19348) — Algorithm 1、§2.4、§3.4.1、§4 超参
- Muon 原论文：[Momentum Orthogonalized by Newton-Schulz](https://arxiv.org/abs/2502.16982)
- Kimi Muon 实践（V4 引用）：Moonshot Kimi 技术报告中的 Muon + QK-Clip（V4 省略 QK-Clip）