# MSLoc 项目理解与 X-CLIP 替代视频编码器调研

调研日期：2026-09-14

## 1. 项目主线

这个仓库由两个串联阶段组成：

1. `DeMamba/` 把长视频切成 2 秒窗口，每窗采样 8 帧，预测 `real / fake / real-to-fake / fake-to-real` 四类状态，再把连续伪造窗口合并成时间 proposal。
2. `Trace/` 以第一阶段 proposal 为条件，让多模态大模型完成定位、解释和 proposal 优化；当前训练路线是 SFT -> OPD -> GRPO。

第一阶段又有两条实验线：

- 全特征 baseline：取视觉骨干最后一层所有 patch token，经逐空间位置的双向 Mamba 做 8 帧时序融合。
- 神经元方法：在真实/伪造配对帧上按层探测敏感通道，从 12 层累计选择恰好 768 个通道，再送入同一个 Mamba 和分类头。

一个重要事实是：当前 `XCLIPVisionModel` 在代码里先把 `[B,T,3,H,W]` 展平为 `[B*T,3,H,W]`，逐帧编码；真正的跨帧建模发生在后面的 DeMamba，而不是 X-CLIP 内部。因此当前所谓 X-CLIP 视频编码器，实际上是 X-CLIP checkpoint 的图像视觉塔。

仓库已经实现 DINOv2-B/14 和 DINOv3-B/16 的全 patch-token及神经元版本，但它们同样逐帧编码，不属于真正的时空视频骨干。

## 2. 选择标准

候选模型按以下顺序评估：

1. 是否原生支持 8 帧；
2. 是否在视频深伪、AI 生成视频检测或时间伪造定位中有直接使用证据；
3. 是否输出或可截取 patch token 与多层 hidden states，以保留神经元探测；
4. 与当前 `8 x 224 x 224`、`196 patch x 768 channel` 接口的兼容度；
5. 权重、代码、显存与训练复杂度。

“更强”不能仅凭不同数据集上的数字证明。下表的排序表示对本项目的预期研究价值和接入性，不声称在 TASLE/ActivityForensics 上已经有同设置的直接胜负。

## 3. 候选排序

| 优先级 | 候选 | 8 帧 | 视频鉴伪先例 | 与现有接口 | 判断 |
|---|---|---:|---|---|---|
| 1 | UniFormerV2-B/16 | 原生 | GenVidBench 将其作为 AI 生成视频检测基线 | 极高：224、patch16、width768；需适配真正的时空 token | 最适合作为第一组替换实验 |
| 2 | InternVideo2-B/L 或 1B-CLIP-f8 | 原生 | 新近 EA-Swin 证明强视频表征对 AIGV 检测有效；但直接使用 InternVideo2 的公开鉴伪证据较少 | 中：8×224、3D token；模型较重 | 最高上限路线，建议在小模型验证后进行 |
| 3 | CLIP ViT-B/16 + ST-Adapter（DeepShield 路线） | 可直接设 8；论文用 12 | DeepShield 直接用于跨数据集人脸深伪检测 | 最高：保留 196×768逐帧 patch 和逐层通道 | 若必须保留当前神经元探测定义，这是最稳妥路线 |
| 4 | DeCoF：CLIP ViT-L/14 + 2-layer temporal Transformer | 原生 8 | 专门用于 AI 生成视频检测，公开代码明确均匀取 8 帧 | 中低：256 patch、width1024，需改 Mamba 与分类头 | 简单、鉴伪匹配强，但不是原位替换 |
| 5 | VideoMAE-Base/Large | 标准多为 16；8 帧需插值/重配时序位置 | DeepfakeBench、WAFL 时间伪造定位、EVAS 等直接采用 | 中：tubelet 会把时间维压缩；需重写 token 布局 | 鉴伪文献证据最丰富，但不满足“原生 8 帧最好” |
| 6 | V-JEPA 2 + EA-Swin | 常用 32/64；EA-Swin 用 32 帧输入、16 个时序输出 | EA-Swin 专门做现代 AIGV 检测且消融中 V-JEPA2 最优 | 低 | 很强的后续上限实验，不适合当前 8 帧第一步 |

## 4. 推荐结论

### 路线 A：先做 UniFormerV2-B/16（首选）

它是最干净的 8 帧原生替代：官方 MMAction2 配置明确 `num_frames=8`，输入 224，patch 16，width 768，12 层。空间网格仍是 14×14，因而当前分类头的大部分维度假设可以保留。它还在大规模 AI 生成视频检测基准 GenVidBench 中被作为视频 Transformer 基线，而不是只在动作识别中出现。

需要注意：不能只取 UniFormerV2 的最终视频级向量，否则会破坏本项目“每个空间 patch 沿 8 帧做 Mamba”和逐层神经元选择的实验定义。应从 backbone 的中间块截取保留时间轴的 token，先实现 full-feature baseline，再决定如何定义跨层神经元。

### 路线 B：保留神经元论文叙事时，用 ST-Adapter

DeepShield 在 CLIP ViT-B/16 的每个 Transformer block 中加入轻量时空 adapter，并继续输出每帧 CLS 与 patch embedding。它与当前 X-CLIP B/16 的层数、宽度、patch 网格最接近，几乎不改变探测器的坐标系。虽然 DeepShield 原文使用 12 帧，adapter 本身不要求固定 12 帧，改成 8 帧比改造 VideoMAE 的 tubelet token 更自然。

严格说这不是“换一个更大的编码器”，而是把当前逐帧视觉塔升级为真正感知时间的鉴伪编码器；对现有方法学最友好。

### 路线 C：第二轮再上 InternVideo2

InternVideo2 的论文明确稀疏采样 8 帧，使用 3D 位置编码，并在多项视频理解、时间定位任务上强于 CLIP/CLIP+SlowFast。小型 B/L 版本比 1B/6B 更适合作为项目候选。它的表征上限很高，但时空 token 与现有逐帧 patch 语义不同，神经元探测、Mamba输入和显存策略都要重新设计。

### 为什么不先做 VideoMAE

VideoMAE 是“别人做视频鉴伪用得最多”的安全答案：DeepfakeBench 已有 detector，WAFL 直接把 VideoMAE 作为时间伪造定位视觉编码器，更新的多模态定位工作也继续采用它。但官方 VideoMAE/VideoMAEv2 默认通常是 16 帧、tubelet size 2。8 帧虽然可以运行，却涉及时间位置编码和预训练分布变化；同时输出只有 4 个 tubelet 时间步，会改变当前逐帧 8 步 Mamba 的含义。

## 5. 建议实验矩阵

为避免把“骨干更强”和“头部结构变化”混在一起，建议先只跑全特征 baseline：

| 实验 | 编码器 | 时序头 | 帧数 | 目的 |
|---|---|---|---:|---|
| E0 | 当前 XCLIPVision-B/16 | 当前 DeMamba | 8 | 原始基线 |
| E1 | UniFormerV2-B/16 中间 token | 相同 DeMamba 或 identity head 二选一并分别报告 | 8 | 验证原生视频预训练收益 |
| E2 | XCLIP/CLIP-B16 + ST-Adapter | 当前 DeMamba | 8 | 验证轻量时空适配收益 |
| E3 | InternVideo2-B/L 中间 token | 轻量 pooling/head | 8 | 验证基础视频模型上限 |
| E4 | VideoMAE-B | 轻量 head | 8 与 16 | 分离帧数和 backbone 效果 |

统一报告 TASLE 的 Det_Acc/F1Det/F1Loc，以及 ActivityForensics 的 all、直接生成器重叠、直接生成器未见和逐生成器 AP/AR。边界指标要保持 2 秒窗口与 proposal 合并规则不变。

在 E1 胜过 E0 后，再重做对应骨干的 layer-wise fake/real probe；否则没有必要先投入神经元选择工程。

## 6. 主要来源

- [UniFormerV2 官方代码与模型](https://github.com/OpenGVLab/UniFormerV2)
- [MMAction2 的 UniFormerV2 8 帧配置](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_pre_u8_kinetics710-rgb.py)
- [InternVideo2 论文](https://arxiv.org/abs/2403.15377)
- [InternVideo2 官方代码](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2)
- [DeCoF：Detecting AI-Generated Video via Frame Consistency](https://arxiv.org/abs/2402.02085)
- [DeCoF 官方代码](https://github.com/wuwuwuyue/DeCoF)
- [DeepShield 论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Cai_DeepShield_Fortifying_Deepfake_Video_Detection_with_Local_and_Global_Forgery_ICCV_2025_paper.pdf)
- [DeepShield 官方代码](https://github.com/lijichang/DeepShield)
- [DeepfakeBench 的 VideoMAE detector](https://github.com/SCLBD/DeepfakeBench/blob/main/training/detectors/videomae_detector.py)
- [WAFL：VideoMAE 用于时间伪造定位](https://github.com/wangty1/WAFL)
- [VideoMAEv2 官方代码](https://github.com/OpenGVLab/VideoMAEv2)
- [EA-Swin：V-JEPA2 用于 AI 生成视频检测](https://arxiv.org/abs/2602.17260)
- [V-JEPA 2 论文](https://arxiv.org/abs/2506.09985)
- [ActivityForensics 论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Bao_ActivityForensics_A_Comprehensive_Benchmark_for_Localizing_Manipulated_Activity_in_Videos_CVPR_2026_paper.pdf)
- [ActivityForensics 官方代码](https://github.com/ActivityForensics/activityforensics)

## 7. 调研范围说明

本次通过 Exa 执行了四个检索角度：视频鉴伪骨干、8 帧原生模型、时间伪造定位、现代 AI 生成视频检测，并进一步读取论文与官方仓库。检索覆盖约 200 条返回结果，最终只保留论文、作者官方代码和主流基准实现。没有找到在完全相同 TASLE/ActivityForensics、8 帧、相同训练策略下直接证明某候选必然超过当前 X-CLIP 的公开实验，因此最终建议必须通过上述受控消融确认。
