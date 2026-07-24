# Raschka 解读梗概：DeepSeek V3 → V3.2

> [← 报告目录](./README.md) · [全文解析（含表格）](./raschka-technical-deepseek-v3-v32.md) · [原文](https://magazine.sebastianraschka.com/p/technical-deepseek)
> 作者：Sebastian Raschka · 2025-12-03（更新 2026-01-01）
> 本地对照：[V3 梗概](../versions/v3.md) · [V3.1 梗概](../versions/v3-1.md) · [V3.2 梗概](../versions/v3-2.md) · [MLA](../versions/mla-latent-attention.md) · **[DSA 梗概](../versions/dsa-sparse-attention.md)** · [DSA 逻辑](../dsa/dsa-logic.md) · [RLVR](../versions/rlvr.md)

---

## 一句话

V3.2 = **V3.1-Terminus 权重架构 + DSA 稀疏注意力**；训练侧在 R1 的 **RLVR+GRPO** 上叠加 **DeepSeekMath V2 的自验证** 与一组 **更保守的 GRPO 稳定性补丁**；V3.2-Exp 的首要角色是 **给 V3.2 铺推理 infra**。

---

## 发布时间线

| 时间 | 模型 | 要点 |
|------|------|------|
| 2024-12 | **V3**、**R1** | 同架构；R1 靠 RLVR 出圈 |
| 2025 | **R1-0528** | 架构不变，后训练小升级 |
| 2025 | **V3.1** → **V3.1-Terminus** | **Hybrid** 单模型双模式；Terminus = 小改进 + 128K，**V3.2 续训起点** |
| 2025-09 | **V3.2-Exp** | 在 Terminus 上 **续训加 DSA**；benchmark 不惊艳，但 **铺生态/推理栈** |
| 2025-11-27 | **DeepSeekMath V2** | 自验证 + 自 refine；V3.2 数学奖励的数据与方法来源 |
| 2025-12-01 | **V3.2** | 架构同 Exp；完整后训练成品 |
| 2025-12-31 | **mHC 论文** | 残差路径研究（Hyper-Connections 约束流形），**非 V3.2 权重变更** |

---

## 架构演进

| 版本 | 架构相对 V3 | 训练侧重 |
|------|-------------|----------|
| V3 | MoE + **MLA** 基座 | 预训练 base |
| R1 | **同 V3** | **RLVR + GRPO**（可验证奖励） |
| V3.1 / Terminus | **同 V3** | Hybrid chat + reasoning（prompt 切换） |
| V3.2-Exp / V3.2 | **+ DSA**（lightning indexer + top-$k$=2048） | Exp 验稀疏；V3.2 全任务 + 工具/agent |

**DSA 复杂度**：注意力主路径 $O(L^2) \to O(Lk)$，$k \ll L$。目标 **不是刷过 Terminus benchmark**，而是 **少掉点、大幅省算**。

---

## 模型类型：专用推理 vs Hybrid

| 路线 | 代表 | Raschka 观察 |
|------|------|----------------|
| 专用推理 | R1、部分 Qwen3 拆分版 | 单场景更强，但要维护两套模型 |
| **Hybrid** | Qwen3 初版、gpt-oss、**V3.1/V3.2** | 一套权重，模板/系统提示切换模式 |
| DeepSeek 方向 | R1（专用）→ **V3.1+ Hybrid** | R1 像推理方法试验床；V3.2 面向通用旗舰 |

---

## MLA 要点

- K/V **压进 latent** 再进 KV cache；推理时 **up-project** 回注意力空间（类比 LoRA 升降维）。
- Q 在训练时也压缩，**推理时不压缩**（文章 side note）。
- MLA 自 **V2** 引入，V3/R1 沿用。

---

## RLVR / GRPO 要点

### R1 奖励

| 奖励 | 作用 |
|------|------|
| format | 答案格式 |
| language consistency | 避免中英混杂 |
| verifier | 数学/代码对错（符号验证） |

### V3.2 奖励

| 任务类型 | 奖励组成 |
|----------|----------|
| reasoning / agent | rule-based outcome + **length penalty** + language consistency |
| general | **生成式 RM**（每 prompt 自带 rubric） |
| math | 并入 **DeepSeekMath V2** 数据与奖励 |

→ 从「纯 RLVR」变为 **RLVR + LLM-as-judge 混合 pipeline**。

### V3.2 GRPO 相对 DAPO / Dr. GRPO

| 改动 | 摘要 |
|------|------|
| **保留** 原 GRPO advantage 归一化 | 不像 Dr. GRPO / DAPO 那样大改目标 |
| **分域 KL 权重** | 数学可近零 KL，但不全局删除 |
| **无偏 KL 估计** | KL 项用与主 loss 相同的 importance ratio 重加权 |
| **off-policy sequence masking** | 负 advantage 且偏离 rollout 太远的序列丢弃 |
| **MoE routing 固定** | rollout 专家路径训练时复现 |
| **top-p/k mask 保留** | 训练 action space 与采样一致 |

---

## DeepSeekMath V2 → V3.2 数学能力

| 组件 | 角色 |
|------|------|
| **LLM1** 证明生成器 | 主模型 |
| **LLM2** 证明验证器 | 训练时作 reward model；rubric 0 / 0.5 / 1 |
| **LLM3** meta-verifier | 训练时监督 LLM2；推理不用 |
| **自 refine** | 最多 8 轮；训练用独立 verifier，**推理用单一 2-in-1 模型** |

**动机**：RLVR 只验最终答案 → 推理过程可错；定理证明需要 **过程分** 而非数值答案。

---

## V3.2 变体

| 模型 | 特点 |
|------|------|
| **V3.2** | 通用 + agent + 工具；DSA + 混合奖励 |
| **V3.2-Speciale** | RL 阶段 **仅 reasoning 数据**；减弱 length penalty → 更长链、更高精度（推理 scaling） |

---

## 结论表

| # | takeaway |
|---|-----------|
| 1 | V3.2 架构与 V3 系一脉；**主要结构改动 = V3.2-Exp 的 DSA** |
| 2 | 数学靠 **DeepSeekMath V2 自验证** 迁入训练 |
| 3 | **GRPO 稳定性补丁** 多于算法替换 |
| 4 | 文章未展开：蒸馏、长上下文训练、工具集成（类 gpt-oss） |
| 5 | 附录 **mHC**：在 Hyper-Connections 上做流形约束，改 **残差路径** 稳定性（2025-12-31 论文） |

---

## 与本地文档映射

| 文章章节 | 本地延伸 |
|----------|----------|
| §3.1 MLA / V3 | [MLA 低秩注意力](../versions/mla-latent-attention.md) · [DeepSeek-V3](../versions/v3.md) |
| §3.2 RLVR / GRPO | [RLVR](../versions/rlvr.md) · [DeepSeek-R1](../versions/r1.md) |
| §3.4 V3.1 Hybrid / MLA 切换 | [DeepSeek-V3.1 梗概§MLA 模式切换](../versions/v3-1.md#mla-模式切换terminus-起) |
| §4 DSA | **[Lightning Indexer 详解](../dsa/lightning-indexer.md)** · [DSA 稀疏注意力](../versions/dsa-sparse-attention.md) · [DSA 逻辑详解](../dsa/dsa-logic.md) |
| §6 V3.2 infra | [DeepSeek-V3.2](../versions/v3-2.md) · [演进总览 §3.6](./deepseek-version-lineage-20260625.md#36-deepseek-v32--v32-exp) |
| §8 mHC | [mHC](../versions/mhc-manifold-hyper-connections.md) · [DeepSeek-V4](../versions/v4.md) |
