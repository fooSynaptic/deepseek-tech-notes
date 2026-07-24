# MLA前向计算流程

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-mechanism-atlas) · [← 演进总览 §2](../01-总览/01-版本演进总览.md#2-版本时间线与关系) · [← 算法线导读](../01-总览/05-算法线导读.md) · [← 基础设施线导读](../01-总览/06-基础设施线导读.md) · [← V3 梗概](01-V3基座.md) · [V2 MLA 首发](../04-版本代际/00-V2-MoE与MLA.md) · [V3.1 Hybrid MLA 切换](../04-版本代际/01-V3.1-Terminus.md#mla-模式切换terminus-起) · [下游 DSA](../05-DSA稀疏注意力/02-DSA梗概.md) · [Raschka §3.1 MLA](../08-外部解读/01-Raschka要点速读.md#mla-要点31)
> **论文**：DeepSeek-V2 首次提出 MLA [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)；V3/R1/V3.1/V3.2 **沿用同一 MLA 结构**

## 核心结论摘要

- MLA 将 K/V **压入低维 latent** 再写 cache，推理时升维得到多头 K/V。
- RoPE 的 R 分量维度小且 **全头共享**，进一步省 cache。
- 单 token cache 从 O(n_h d_h) 降至 **O(d_c^KV + d_h^R)**，约 MHA 的 1/57。
- V2 首发，V3/R1/V3.1/V3.2 **沿用同一 MLA 结构**。

---

## 一句话

MLA 把 K/V **先压到低维 latent $c^{KV}$ 再写入 KV cache**；推理时对 latent **升维** 得到多头 K/V 的 content 部分；**RoPE 的 R 分量**维度小且 **全头共享**。相对标准 MHA，单 token cache 可从 $O(n_h d_h)$ 降到 **$O(d_c^{KV} + d_h^R)$**。

---

<a id="forward-flow"></a>

## 流程图

<img src="figures/mla-forward-flow.svg" alt="MLA 前向计算流程（公式 37–47）：Query / Key / Value 三分支、KV Cache 压缩对比（DeepSeek-V2 维数标注）" width="920"/>

[图示详情](figures/mla-forward-flow.svg)

> **为什么 1536 能变成 [128,128] 和 [128,64]?** ——不是切分,是两个独立上投影矩阵放大后按头 reshape:
>
> - $q_t^C = W^{UQ} c_t^Q$: $[16384 \times 1536] \cdot [1536] \to [16384]$, 其中 $16384 = n_h \times d_h = 128 \times 128$ → reshape $[128, 128]$
> - $q_t^R = \mathrm{RoPE}(W^{QR} c_t^Q)$: $[8192 \times 1536] \cdot [1536] \to [8192]$, 其中 $8192 = n_h \times d_h^R = 128 \times 64$ → reshape $[128, 64]$
>
> $[128,128]$ 里两个 128 含义不同:前一个是头数 $n_h$(共 128 个头),后一个是每头维度 $d_h$(每头 128 维),本配置恰好都等于 128。二者都是架构超参,不是从 1536 算出来的;1536 只决定矩阵的列数。(KV 侧同理:$c_t^{KV} = 512$ 经 $W^{UK}, W^{UV}$ 投影成 $[128,128]$。)

> **右边 $k_t^R = [64]$ 的 64 怎么来?** ——$64 = d_h^R$(每头 RoPE 维度,架构超参);$W^{KR}: [64 \times 5120] \cdot h_t \to [64]$,再加上 RoPE。
> **关键:** $k_t^R$ 没有头维度——所有 $n_h = 128$ 个头共享同一个 $[64]$(解耦 RoPE);而左边 $q_t^R$ 是每头各一份 $[128, 64]$。
> 正因为 K 的 RoPE 部分全局只存一份 $[64]$(不按头复制),KV 缓存才这么小——这是 MLA 省显存的另一半原因。

> **MLA 到底压缩了谁?如果不做压缩会变多大?** ——下面三项就是 MLA 压缩/解耦的对象(格式:MLA 压缩后 $\Rightarrow$ 不压缩):
>
> - $c_t^Q$ 查询潜向量: $1536 \Rightarrow 16384\ (= n_h d_h)$, 约 11×; 不进缓存,省的是参数与计算量。
> - $c_t^{KV}$ KV 联合潜向量: $512 \Rightarrow 16384\ (= n_h d_h)$, 32×; ★进缓存 —— 这是省显存的核心。
> - $k_t^R$ 共享 RoPE 键: $64 \Rightarrow 8192\ (= n_h d_h^R)$, 128×; ★进缓存,靠全头共享(不按头复制),而非低秩压缩。
>
> 缓存总量: 标准 MHA $= 2n_h d_h = 32768$ → MLA 若不压缩 $= 16384 + 64 = 16448$(仅 MHA 一半) → 实际 MLA $= 512 + 64 = 576 \approx$ MHA 的 1/57

**口诀**：$h_t$ → Q 降维 $c^Q$ / KV **共享**降维 $c^{KV}$ → Q/K 拆头 C + 共享 R（RoPE）→ V 仅 C → 注意力 → concat + $W^O$。

---

## 符号

| 符号 | 含义 | 典型值（V2 论文） |
|------|------|-------------------|
| $d$ / $d_{\mathrm{model}}$ | 隐状态维度 | 5120 |
| $n_h$ | 注意力头数 | 128 |
| $d_h^C$ | 单头 content 维 | 128 |
| $d_h^R$ | RoPE 维（全头共享） | 64 |
| $d_h$ | 单头 Q/K 维 $d_h^C + d_h^R$ | 192 |
| $d_c'$ | Q 侧 latent 秩 | 1536 |
| $d_c$ / $d_{\mathrm{latent}}^{KV}$ | **KV 共享** latent 秩 | **512** |

---

## 三分支计算

### ① Query

| 步 | 公式 | Shape |
|----|------|-------|
| 压缩 | $c_t^Q = W^{DQ} h_t$ | $[d] \to [d_c']$ |
| 升维 C | $q_t^C = W^{UQ} c_t^Q$ → 按头拆 $q_{t,i}^C$ | $[n_h \cdot d_h^C]$ |
| RoPE R | $q_t^R = \mathrm{RoPE}(W^{QR} c_t^Q)$ | $[d_h^R]$，**所有头共用** |
| 拼接 | $q_{t,i} = [q_{t,i}^C;\, q_t^R]$ | $[d_h]$ |

### ② Key

| 步 | 公式 | Shape |
|----|------|-------|
| **KV 共享压缩** | $c_t^{KV} = W^{DKV} h_t$ | $[d_c]$ — **K 与 V 共用** |
| 升维 C | $k_t^C = W^{UK} c_t^{KV}$ → $k_{t,i}^C$ | 按头拆 |
| RoPE R | $k_t^R = \mathrm{RoPE}(W^{KR} h_t)$ | 来自 **$h_t$**（非 $c^{KV}$），全头共享 |
| 拼接 | $k_{t,i} = [k_{t,i}^C;\, k_t^R]$ | $[d_h]$ |

### ③ Value

| 步 | 公式 | 说明 |
|----|------|------|
| 复用 latent | 同 $c_t^{KV}$ | 与 Key 分支共享降维结果 |
| 升维 | $v_t^C = W^{UV} c_t^{KV}$ → $v_{t,i}^C$ | **无 RoPE、无 R 支路** |

### ④ 注意力与输出

$$
o_{t,i} = \sum_{j=1}^{t} \mathrm{Softmax}_j\left( \frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h^C + d_h^R}} \right) v_{j,i}^C
$$

$$
u_t = W^O\, [o_{t,1};\ldots;o_{t,n_h}]
$$

---

## KV Cache 里到底存什么

| | **标准 MHA** | **MLA（推理）** |
|--|--------------|-----------------|
| 每 token 缓存 | $n_h$ 份 K + $n_h$ 份 V | **$c^{KV}$** + **$k^R$**（共享） |
| 典型字节量（上表维度） | $2 \times n_h \times d_h^C \approx 32768$ 维量级 | $d_c + d_h^R = 512 + 64 = 576$ |
| 压缩比 | 1× | **约 1/57** |

推理时从 cache 读出 $c_j^{KV}$ 再乘 $W^{UK}$、$W^{UV}$ **现场升维**；多一次矩阵乘，换显存。

> V3.1 **Hybrid**：Prefill 时 Q/K[可按 **MHA 式**展开 latent；Decode 时 **MQA 式**共享 latent](../04-版本代际/01-V3.1-Terminus.md#mla-模式切换terminus-起)。**cache 布局仍是 MLA latent**，切换的是展开方式。

---

## 基础设施线位置

| 方向 | 文档 |
|------|------|
| **本节点（① 同质 MLA KV）** | [基础设施线导读 §1](../01-总览/06-基础设施线导读.md#1-演进链kv--offload) |
| **下游 ② 异构 cache** | [DSA稀疏注意力§异构 KV](../05-DSA稀疏注意力/02-DSA梗概.md#异构-kv-cache) |

---

## 算法线位置

| 方向 | 文档 |
|------|------|
| **本节点（① MLA）** | [算法线导读 §1](../01-总览/05-算法线导读.md#1-演进链attention--残差) |
| **下游 ② DSA** | [DSA 稀疏注意力](../05-DSA稀疏注意力/02-DSA梗概.md)（MLA 结构不变，加稀疏选择） |
| **下游 ③ CSA/HCA** | [CSA / HCA](../04-版本代际/05-CSA-HCA混合压缩注意力.md) · [DeepSeek-V4](../04-版本代际/03-V4.md)（新注意力，不再单一 MLA latent） |

---

## 与后续版本

| 版本 | MLA 变化 |
|------|----------|
| V3 / R1 / V3.1 | 稠密 MLA attention |
| **V3.2 + DSA** | MLA **不变**；[indexer 在 latent 序列上选 top-$k$](../05-DSA稀疏注意力/02-DSA梗概.md)、[Core 仍做 MLA](../05-DSA稀疏注意力/02-DSA梗概.md) |
| **V4** | CSA/HCA 等 **[新注意力](../04-版本代际/05-CSA-HCA混合压缩注意力.md)**，不再单一 MLA latent · [DeepSeek-V4](../04-版本代际/03-V4.md) |

---

## 延伸

| 资源 | 说明 |
|------|------|
| [MLA前向计算流程（含 PyTorch 对照）](02-MLA低秩注意力.md) | 更长的 shape 推演与 `mla_forward.py` |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | V3+ 推理 kernel |
| [DeepSeek-V3](01-V3基座.md) | V3 梗概中的 MLA 一行 |

**论文**：V2 [2405.04434](https://arxiv.org/abs/2405.04434) · V3 [2412.19437](https://arxiv.org/abs/2412.19437)