# TRACE 取证文字解释奖励：调研、设计与实现

## 1. 范围

完整流程保持为：

```text
DeMamba proposals → ref2 SFT → paired-teacher precheck → OPD → GRPO → test
```

本次只替换 GRPO 的文字解释奖励。定位奖励、格式奖励、SFT 和 OPD 均不改变；OPD 仍只蒸馏 paired teacher 的事件/时间结构，不蒸馏自由解释文本。

## 2. 参考工作

- CIDEr 与 SCST：caption 模型可直接把不可微的整句指标作为 RL reward，不需要生成式裁判。[1][2]
- SPICE：把参考与预测 caption 解析成对象、属性、关系 tuple，并计算 Precision、Recall、F1；这是本实现“样本级证据图”的直接依据。[3]
- SPIDEr：联合 CIDEr 与 SPICE；SPICE 官方实现也指出，单独优化图召回会产生重复和不流畅文本，因此必须惩罚重复并兼顾 Precision。[4]
- CIDEnt：在视频 caption reward 中加入 entailment，避免词面相似但逻辑矛盾的文本获奖。[5]
- FakeReasoning：把鉴伪解释组织成固定 forgery attributes，说明结构化取证属性比开放式整体裁判更容易监督。[6]

这些工作共同支持：有可信参考解释时，先奖励可核验的取证事实，不必为每条 rollout 调用 235B VLM。

## 3. 设计

TASLE annotation 已包含：

```text
object_class / object_caption
start_class  / start_caption
end_class    / end_caption
```

实现将其视为每条样本的小型 Forensic Evidence Graph：

```text
(event, object_anomaly, object evidence)
(event, onset, start-boundary evidence)
(event, offset, end-boundary evidence)
```

其中核心对象/异常权重为 2，开始和结束证据权重各为 1。生成解释按句号、分号等切成 claims；每条 claim 与每个参考事实计算匹配分。

解释奖励为：

\[
R_{text}=0.5R_{recall}+0.5R_{precision}
-0.5R_{contradiction}-0.2R_{generic}
-0.1R_{repeat}-0.1R_{length}.
\]

- `recall`：标准答案的重要要点说全没有；
- `precision`：生成内容能否由标准答案支持，防止堆砌 artifact；
- `contradiction`：与参考事实矛盾；
- `generic`：只说“视频是假的”等空话；
- `repeat`：重复短语；
- `length`：超过合理长度。

结果裁剪到 `[-1,1]`。现有定位 IoU gate 仍然生效：定位不正确时不发放文字解释奖励。

## 4. 两种模式

### lexical

使用规范化、领域同义词和 token F1。无需新模型、无需训练、速度最快，适合 smoke 和基线。

### nli

在 lexical 分数上增加冻结的 `cross-encoder/nli-deberta-v3-small`，识别同义改写和 contradiction。模型在 GRPO 启动时加载一次，本次不微调 NLI。它只读取参考文字与生成解释，不读取视频。

## 5. 代码位置

- `trace/text_explanation_reward.py`：证据图、claim 切分、lexical/NLI 匹配和最终 reward；
- `trace/train_mt.py`：加载本地 scorer；
- `trace/opd_grpo.py`：保留三类 reward 接口并记录解释分量；
- `trace/trace_trainer.py`：记录 graph Precision/Recall/F1、contradiction 和反投机指标；
- `scripts/train/grpo.sh`：选择 `TEXT_REWARD_MODE=lexical|nli`；
- `tests/test_opd_grpo.py`：本地文本 reward 的静态单元测试。

原 Qwen3-VL-235B 在线 judge 已退出训练路径。

## 6. 验证建议

依次比较：

1. OPD + GRPO，仅定位/格式；
2. 加 lexical explanation reward；
3. 加 NLI explanation reward。

人工检查同义改写、对象替换、否定反转、artifact 堆砌和重复五类样本。自动 reward 上升但人工判断不升时，应视为 reward hacking。

## References

[1] [Vedantam, Zitnick, Parikh, “CIDEr: Consensus-Based Image Description Evaluation,” CVPR, 2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.html).

[2] [Rennie et al., “Self-Critical Sequence Training for Image Captioning,” CVPR, 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.html)；[官方实现](https://github.com/ruotianluo/self-critical.pytorch)。

[3] [Anderson et al., “SPICE: Semantic Propositional Image Caption Evaluation,” ECCV, 2016](https://arxiv.org/abs/1607.08822)；[官方实现](https://github.com/peteanderson80/SPICE)。

[4] [Liu et al., “Improved Image Captioning via Policy Gradient Optimization of SPIDEr,” ICCV, 2017](https://openaccess.thecvf.com/content_ICCV_2017/html/Liu_Improved_Image_Captioning_ICCV_2017_paper.html).

[5] [Pasunuru, Bansal, “Reinforced Video Captioning with Entailment Rewards,” EMNLP, 2017](https://aclanthology.org/D17-1103/).

[6] [Gao et al., “Toward Generalizable Forgery Detection and Reasoning,” IEEE TIP, 2026](https://github.com/PRIS-CV/FakeReasoning)（官方代码与数据说明）。
