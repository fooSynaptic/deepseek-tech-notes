# DeepSeek 各版本梗概

> 更新：2026-06-25
> [← 中文导读](../README.md) · [← 仓库首页（EN）](../../README.md) · [版本演进总览](../reports/deepseek-version-lineage-20260625.md) · **[算法线导读](../reports/deepseek-algorithm-line.md)** · **[基础设施线导读](../reports/deepseek-infra-line.md)** · **[MoE 线导读](../reports/deepseek-moe-line.md)**

每篇一页纸梗概：定位、核心改动、infra 关注点、上下游关系。

| **算法线** | [算法线导读](../reports/deepseek-algorithm-line.md) | MLA → DSA → CSA/HCA + mHC 专题 hub |
| **基础设施线** | [基础设施线导读](../reports/deepseek-infra-line.md) | MLA KV → 异构 Cache → Index Share → ESS → V4 HiSparse |
| **MoE 线** | [MoE 线导读](../reports/deepseek-moe-line.md) | 稠密 FFN → DeepSeekMoE → aux-loss-free → Hash MoE |

| 版本 | 文档 | 一句话 |
|------|------|--------|
| **V1** | [DeepSeek-LLM V1](./v1.md) | DeepSeek-LLM 机制总结（7B/67B；Figure 2–5 / Table 3–4） |
| **V1 BBPE** | [V1 BBPE 词表](./v1-bbpe-tokenizer.md) | Byte-level BPE 词表、预分词规则、102,400 embedding |
| **V2** | [DeepSeek-V2](./v2.md) | 236B/21B；**MLA + DeepSeekMoE** 首次引入；128K |
| **V1→V3** | [V1→V3 前代演进](../reports/deepseek-v1-to-v3-lineage.md) | 前代三代对照与演进逻辑 |
| **V3** | [DeepSeek-V3](./v3.md) | 671B MoE + MLA 基座，开源旗舰起点 |
| **V3 FP8** | [V3 FP8 动态量化](./v3-fp8-dynamic-quantization.md) | 训练侧 FP8 块级动态 scale + FP32 累加提升 |
| **R1** | [DeepSeek-R1](./r1.md) | V3-Base + [RLVR](./rlvr.md)；架构不变 |
| **RLVR** | [RLVR](./rlvr.md) | 可验证奖励 + GRPO；R1 后训练核心 |
| **MLA** | [MLA 低秩注意力](./mla-latent-attention.md) | latent 压缩 KV；[前向流程图](./mla-latent-attention.md#forward-flow)（Eq. 37–47） |
| **DeepSeekMoE** | [DeepSeekMoE](./deepseek-moe.md) | 细粒度 routed + shared；V2 首发、V3 旗舰化 |
| **MoE 路由** | [aux-loss-free MoE 路由](./aux-loss-free-moe-routing.md) | aux-loss-free 动态 bias 负载均衡（V3 论文 Table 5） |
| **Seq-wise $L_{\mathrm{Bal}}$** | [序列均衡损失](./moe-sequence-wise-balance-loss.md) | 单序列内 $f_i P_i$ 互补均衡（Eq. 17–20） |
| **V3.1** | [DeepSeek-V3.1](./v3-1.md) | Hybrid 推理，无架构变更，128K |
| **V3.2** | [DeepSeek-V3.2](./v3-2.md) | DSA 稀疏注意力，长上下文效率拐点 |
| **DSA** | [DSA 稀疏注意力](./dsa-sparse-attention.md) | indexer + top-$k$ + Core MLA；[完整逻辑](../dsa/dsa-logic.md) |
| **Index Share** | [Index Share 梗概](./index-share.md) | IndexCache 纯 infra 补丁，社区称「V3.3」 |
| **ESS** | [ESS Latent offload](./ess-latent-cache-offload.md) | Latent-Cache CPU offload；[论文梗概](./ess-paper-highlights.md) |

**推理答疑**：[答疑索引](./qa/README.md)（如 [H2D / D2H](./qa/h2d-d2h-pcie-transfer.md)）

> DSA / Index Share **逻辑详解**：[DeepSeek DSA 与 Index Share 系列](../dsa/README.md)

| **mHC** | [mHC](./mhc-manifold-hyper-connections.md) | 双随机流形约束残差超连接（含 §3 流形推导）；V4 落地 |
| **Hyper-Connections** | [Hyper-Connections](./hyper-connections.md) | $n$ 路并行残差流 + pre/post/comb；mHC 前置（HC 子专文） |
| **CSA / HCA** | [CSA / HCA](./csa-hca-mixed-attention.md) | 4:1 稀疏 + 128:1 dense 混合压缩注意力；V4 算法线 ③ |
| **Hash MoE + FP4** | [Hash MoE + FP4](./hash-moe-fp4.md) | 前几层 Hash 路由 + routed expert FP4；MoE 线 ⑤ |
| **Muon** | [Muon 优化器](./muon-optimizer.md) | 矩阵正交化优化器；V4 训练侧替换大部分 AdamW |
| **V4** | [DeepSeek-V4](./v4.md) | V4-Pro / V4-Flash 梗概，1M context |
| **V4 KV layout** | [V4 KV Layout](./v4-kv-layout.md) | Classical + State 双池 |
| **V4 HiSparse** | [V4 HiSparse](./v4-hisparse.md) | inactive C4 CPU offload；~3× KV 容量 |
| **V4 磁盘 Prefix** | [V4 磁盘 Prefix Cache](./v4-disk-prefix-cache.md) | CSA/HCA 落盘 + SWA 三档策略（§3.5.2） |
| **DSpark / 投机解码** | [投机解码与 DSpark](./dspark-speculative-decoding.md) | **唯一专文**（MTP + 自测 + DSpark + MTP-1） |
