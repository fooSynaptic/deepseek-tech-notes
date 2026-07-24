# Raschka 全文解析：From DeepSeek V3 to V3.2

> [← 梗概](01-Raschka要点速读.md) · [← 报告目录](../01-总览/03-技术报告索引.md)
> **第三方原文**：[From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates](https://magazine.sebastianraschka.com/p/technical-deepseek)（Sebastian Raschka · Ahead of AI · 2025-12-03）
> 本地对照：[V3.1](../04-版本代际/01-V3.1-Terminus.md) · [V3.2](../04-版本代际/02-V3.2-DSA.md) · [DSA 梗概](../05-DSA稀疏注意力/02-DSA梗概.md) · [DSA 逻辑](../05-DSA稀疏注意力/03-DSA逻辑详解.md) · [RLVR](../03-后训练与R1/01-RLVR.md)

---

## 1. The DeepSeek Release Timeline

**要点**：V3（2024-12）起初不火爆；**R1** 使同架构模型成为开源旗舰。2025 年无「隔年大招」，但有 **V3.1 / V3.2-Exp** 等铺垫 release；**V3.2-Exp 意在预热推理 infra**，正式 **V3.2** 随后发布。V3.2 使用 **非标准稀疏注意力**，需自定义推理代码。

### 表 1-1：主要 release 时间线

| 时间 | 模型 / 事件 | 角色（Raschka 归纳） |
|------|-------------|---------------------|
| 2024-12 | DeepSeek **V3** | Base：MoE + MLA |
| 2024-12 ~ 2025-01 | DeepSeek **R1** | 同架构 + RLVR；带火 V3 系 |
| 2025 | **R1-0528** | 后训练小升级，对标 o3 / Gemini 2.5 Pro 时期 |
| 2025 | **V3.1**、**V3.1-Terminus** | Hybrid；Terminus = V3.1 收尾 checkpoint |
| 2025-09 | **V3.2-Exp** | Terminus + **DSA** 续训；benchmark 平淡，**铺生态** |
| 2025-11-27 | **DeepSeekMath V2** | 数学 PoC；自验证 pipeline |
| 2025-12-01 | **V3.2** | 旗舰正式版；架构同 Exp |
| 2025-12-31 | **mHC** 论文 | 残差 Hyper-Connection 研究（附录 §8） |

> 原文 Figure 3：release 时间轴（主模型标红）。团队曾传训练芯片从 NVIDIA 换 Huawei，文章称 **已回到 NVIDIA**（基于公开信息）。

---

## 2. Hybrid Versus Dedicated Reasoning Models

**要点**：V3 = **base**；R1 = **专用推理**（额外 post-training）。行业出现 **Hybrid**（同一 checkpoint 切换 thinking/chat）与 **拆分 instruct/reasoning** 两条路线。

### 表 2-1：推理模型形态对比

| 形态 | 机制 | 代表 |
|------|------|------|
| **Dedicated reasoning** | 独立 checkpoint / pipeline | DeepSeek **R1** |
| **Hybrid** | 模板或系统提示切换模式 | 初版 **Qwen3**、**gpt-oss**、**V3.1 / V3.2** |
| **Hybrid → 拆分** | 分开发布 instruct / reasoning 版 | Qwen 后续路线 |

### 表 2-2：DeepSeek 在「专用 vs Hybrid」上的移动

| 阶段 | 方向 | Raschka 解读 |
|------|------|----------------|
| V3 + R1 | Base + **专用** R1 | R1 验证 RLVR 与推理能力 |
| V3.1 / V3.2 | → **Hybrid** | 单模型覆盖 chat + reasoning；R1 像 **试验床**，V3.2 面向 **通用旗舰** |
| 未来（推测） | 可能仍有 **R2** 专用版 | 文章未证实 |

> 原文 Figure 4–5：R1 训练 pipeline、2025 年 reasoning/hybrid 模型时间线。

---

## 3. From DeepSeek V3 to V3.1

### 3.1 DeepSeek V3 Overview and MLA

**要点**：V3 两大架构亮点 = **MoE** + **MLA**。MLA 将 K/V **投影到低维 latent** 再写入 KV cache；推理时 **up-project** 使用。Q 仅在 **训练** 时压缩，**推理** 不压缩。

### 表 3-1：V3 核心架构组件

| 组件 | 作用 | 备注 |
|------|------|------|
| **DeepSeekMoE** | 条件计算扩容 | 文章略去 MoE 入门 |
| **MLA** | KV **latent 压缩** | V2 引入；V3/R1 沿用；利于 KV cache |
| **MTP** | 推测解码相关 | 本地 [DeepSeek-V3](../02-基座架构/01-V3基座.md) |

> 原文 Figure 6：MLA 降维 → cache → up-project 示意图。

### 3.2 DeepSeek R1 and RLVR

**要点**：R1 **架构同 V3**；差异在 **RLVR**（Reinforcement Learning with Verifiable Rewards）：从可符号/程序验证的任务（数学、代码等）学习。**GRPO** = 无 critic 的 PPO 简化版；RLVR+GRPO **再省掉 reward model**，直接用计算器/编译器等 **可验证奖励**。

### 表 3-2：LLM 强化学习 pipeline 对比

| | RLHF + **PPO** | **GRPO** | **RLVR + GRPO** |
|--|----------------|----------|-----------------|
| Reward model | 人类偏好训练 | — | — |
| Critic（value model） | 有 | **无** | **无** |
| 奖励来源 | RM 打分 | 组内相对优势 | **符号工具**（计算器、编译器等） |
| 典型用途 | 对齐 | 简化 RLHF | **推理**（数学/代码可验证） |

### 3.3 DeepSeek R1-0528

| 项 | 说明 |
|----|------|
| 定位 | 官方称 **minor version upgrade** |
| 架构 | 同 V3/R1 |
| 提升来源 | 后训练 pipeline 优化（细节未公开）；托管版或更长推理 |

### 3.4 DeepSeek V3.1 Hybrid Reasoning

| 项 | 说明 |
|----|------|
| 能力 | **Instruct + reasoning** 合一 |
| 切换 | Chat **prompt 模板**（类初版 Qwen3） |
| 权重链 | V3.1 ← V3.1-Base ← **V3**（**架构相同**） |
| Terminus | V3.1 收尾版；128K；**V3.2 续训基座** → 见 [DeepSeek-V3.1 梗概§MLA 模式切换](../04-版本代际/01-V3.1-Terminus.md#mla-模式切换terminus-起) |

---

## 4. DeepSeek V3.2-Exp and Sparse Attention

**要点**：V3.2-Exp 在 **V3.1-Terminus** 上 **续训加入 DSA**。DSA = **(1) lightning indexer** + **(2) token selector**；从「看全长」变为「看学到的 top-$k$ 子集」。

### 表 4-1：滑动窗口注意力 vs DSA

| | **滑动窗口**（Gemma 3、Olmo 3 等） | **DSA** |
|--|-----------------------------------|---------|
| 可见历史 | **固定宽度** 局部窗 | **学习选择** 的 $k$ 个位置 |
| 选择方式 | 规则（距离） | indexer 打分 + **top-$k$** |
| 稀疏模式 | 带状局部 | 可 **非局部**、数据驱动 |
| 典型 $k$ | 窗口宽 $w$ | **$k=2048$**（官方代码） |

### 表 4-2：Lightning indexer 打分公式符号

公式：

$$
I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}\,\mathrm{ReLU}\!\left(q_{t,j}\cdot k_{s}\right)
$$

| 符号 | 含义 |
|------|------|
| $t$ | 当前 query token 位置 |
| $s$ | 历史 token 位置（$0 \le s < t$） |
| $j$ | indexer head 索引（$1 \ldots H^I$） |
| $q_{t,j}$ | 位置 $t$、head $j$ 的 query 向量 |
| $k_s$ | 位置 $s$ 的 key（**已压缩在 MLA KV cache**） |
| $w_{t,j}$ | 可学习的 per-head 权重 |
| top-$k$ 的 $k$ | **与 key 的 $k$ 无关**；selector 超参，**= 2048** |

**实现注**：

- indexer **只对 query 多头**；keys 已在 cache，无需再按 head 打分。
- ReLU 本身难让分数为 0；**真正稀疏来自 top-$k$ selector**。
- 复杂度：$O(L^2) \to O(Lk)$。

> 原文 Figure 9–11：滑动窗 vs DSA 注意力图、DSA 流程图。
> **本地延伸**：[Lightning Indexer 详解](../05-DSA稀疏注意力/04-Lightning-Indexer详解.md) · [DSA 逻辑详解](../05-DSA稀疏注意力/03-DSA逻辑详解.md)

<img src="../05-DSA稀疏注意力/figures/dsa-pipeline.svg" alt="DSA 两阶段：Lightning Indexer → Top-k → Core MLA；Indexer-Cache 与 Latent-Cache" width="920"/>

---

## 5. DeepSeekMath V2

**时间**：2025-11-27（美国感恩节），V3.2 发布前 4 天。基座：**V3.2-Exp-Base**。角色：V3.2 的 **数学能力 PoC**。

### 表 5-1：常规 RLVR 的局限

| 局限 | 含义 |
|------|------|
| 答案对 ≠ 推理对 | 错误逻辑也可能碰对答案 |
| 定理证明 | 需要 **逐步推导**，**最终数值奖励不适用** |

### 5.1 Self-Verification

**结构**：训练 **证明生成器 LLM1** + **证明验证器 LLM2**；可选 **meta-verifier LLM3** 监督 LLM2。

### 表 5-2：证明验证 rubric

| 分数 | 标准 |
|------|------|
| **1** | 完整严谨，逻辑步骤清晰 |
| **0.5** | 整体逻辑正确，有小错或省略 |
| **0** | 根本性逻辑错误 or 关键缺口 |

### 表 5-3：三模型分工与训练

| 模型 | 基座 / 训练 | 推理时 |
|------|-------------|--------|
| **LLM1** 生成器 | 用 LLM2 作 reward 训练 | 最终 **2-in-1**（生成+自评） |
| **LLM2** 验证器 | V3.2-Exp-SFT + RL（format + 与人类标注分数对齐） | 训练后 **不单独部署** |
| **LLM3** meta-verifier | RL，评估 LLM2 的分析质量 | **仅训练**；meta 分 0.85→0.96（文引数据） |

### 5.2 Self-Refinement

| 模式 | 说明 |
|------|------|
| 经典 self-refinement | **同一 LLM** 生成 → 自评 → 修订 |
| DeepSeek 观察 | 单模型自评易 **盲目宣称正确** |
| 训练时 | 独立 **LLM2**（+LLM3）提供严格反馈 |
| **推理时** | **单一最终生成器** 兼做验证（省算力） |
| 迭代 | 论文最多 **8 轮**；精度随轮次升，**未饱和** |

> 原文 Figure 12–16：generator/verifier/meta-verifier、自 refine 流程、迭代精度曲线。

---

## 6. DeepSeek V3.2

**要点**：对标 GPT-5 / Gemini 3 Pro 级开源旗舰；**架构与 V3.2-Exp 完全相同**（MoE + MLA + **DSA**）；差异在 **训练与后训练**。数学采用 Math V2 pipeline；强调 **工具 / agent**；训练芯片叙述为 **回归 NVIDIA**。

### 6.1 Architecture

| 项 | 内容 |
|----|------|
| 架构声明 | 与 **V3.2-Exp 完全一致** |
| 效率动机 | MLA（V2/V3）+ **DSA** 降长上下文成本 |
| 原文 Figure 19 | DSA 带来的 **推理成本节省**（截图自 V3.2 报告） |

### 6.2 Reinforcement Learning Updates

### 表 6-1：R1 vs V3.2 奖励设计

| | **DeepSeek R1** | **DeepSeek V3.2** |
|--|-----------------|-------------------|
| Format reward | ✅ | ❌ 移除 |
| Language consistency | ✅ | ✅ |
| Verifier / outcome | ✅（数学/代码） | ✅ rule-based outcome（reasoning/agent） |
| Length penalty | — | ✅（**agent** 任务） |
| Generative RM + rubric | — | ✅（**general** 无符号验证任务） |
| Math V2 pipeline | — | ✅ 并入数据集与奖励 |

**归纳**：V3.2 = **RLVR（可验证域）+ 生成式 RM（开放域）** 混合，而非 R1 式纯 verifier RLVR。

### 6.3 GRPO Updates

文章先列 **Olmo 3** 采用的激进 GRPO 改动（DAPO / Dr. GRPO 系），再对比 **V3.2 更保守** 的补丁。

### 表 6-2：Olmo 3 的 GRPO 改动

| 改动 | 来源 |
|------|------|
| Zero gradient signal filtering | DAPO |
| Active sampling（动态采样） | DAPO |
| Token-level loss | DAPO |
| No KL loss | DAPO / Dr. GRPO |
| Clip higher | DAPO |
| Truncated importance sampling | Yao et al. |
| No std normalization in advantage | Dr. GRPO |

### 表 6-3：DeepSeek V3.2 的 GRPO 改动

| 改动 | 说明 |
|------|------|
| **分域 KL 强度** | 保留 KL；数学可 **近零** KL，作超参而非全局删除 |
| **无偏 KL 估计** | KL 项用与主 loss 相同的 **importance ratio** 重加权 |
| **Off-policy sequence masking** | 复用 rollout 时，丢弃 **负 advantage 且过于 off-policy** 的整条序列 |
| **MoE routing 固定** | rollout 记录专家路由，训练时 **强制同路由** |
| **保留 top-p/k 采样 mask** | 训练时 action space 与采样一致 |
| **保留原 GRPO advantage 归一化** | **不**采用 Dr. GRPO / DAPO 的大改归一化 |

**定位**：比 Olmo 3 / DAPO **更接近原始 GRPO**，靠 **工程稳定性补丁** 而非重写目标。

### 6.4 V3.2-Speciale

| 项 | V3.2 | **V3.2-Speciale** |
|----|------|-------------------|
| RL 数据 | 多域 | **仅 reasoning** |
| Length penalty | 常规定 | **减弱** → 更长输出 |
| 行为 | 通用旗舰 | **extended thinking**；更高精度、更多 token（推理 scaling） |

> 原文 Figure 17–18、20：benchmark、架构、Speciale 长度-精度权衡。

---

## 7. Conclusion

| # | Takeaway |
|---|----------|
| 1 | V3.2 架构与 V3 以来一脉相承 |
| 2 | **主要结构变化** = V3.2-Exp 的 **稀疏注意力 DSA** |
| 3 | 数学提升 = 吸收 **DeepSeekMath V2 自验证** |
| 4 | 训练 pipeline 含 **GRPO 稳定性** 等多项更新 |
| 5 | 文章 **未展开**：蒸馏、长上下文训练、工具集成（类 gpt-oss）等 |

---

## 8. Appendix: mHC

**时间**：2025-12-31 论文。焦点从 attention/FFN 转向 **残差路径**。

<a id="表-8-1-transformer-模块演进"></a>

### 表 8-1：Transformer 模块演进

| 模块 | 演进链 |
|------|--------|
| Normalization | LayerNorm → RMSNorm → Dynamic TanH |
| Attention | GQA → [sliding window（SWA）](../04-版本代际/qa/v4-swa-sliding-window.md#行业链) → **MLA** → **sparse (DSA)** |
| FFN | GeLU → SiLU → SwiGLU → **MoE** |
| **残差** | 恒等残差（ResNet）→ **Hyper-Connections (HC)** → **mHC**（流形约束、保范数） |

**mHC**：在 HC（多路并行残差流 + 可学习混合）上，将混合矩阵约束在 **结构化保范数流形**，提升 **训练稳定性**；有少量开销。

> 与 **V3.2 部署权重无直接对应**；在 **V4** 与独立论文 [mHC arXiv:2512.24880](https://arxiv.org/abs/2512.24880) 中落地（本地 [mHC 详解](../04-版本代际/04-mHC流形约束超连接.md) · [DeepSeek-V4](../04-版本代际/03-V4.md)）。

---

## 参考与本地映射

| 资源 | 链接 |
|------|------|
| 原文 | https://magazine.sebastianraschka.com/p/technical-deepseek |
| 梗概 | [Raschka 要点速读](01-Raschka要点速读.md) |
| V3.1 | [DeepSeek-V3.1 梗概](../04-版本代际/01-V3.1-Terminus.md) |
| V3.2 | [DeepSeek-V3.2 梗概](../04-版本代际/02-V3.2-DSA.md) |
| DSA 梗概 | [DSA稀疏注意力](../05-DSA稀疏注意力/02-DSA梗概.md) |
| DSA 逻辑 | [DSA逻辑详解](../05-DSA稀疏注意力/03-DSA逻辑详解.md) |
| RLVR | [RLVR](../03-后训练与R1/01-RLVR.md) |
| 演进总览 | [版本演进总览](../01-总览/01-版本演进总览.md) |