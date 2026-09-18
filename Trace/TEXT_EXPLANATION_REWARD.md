# GRPO 文字解释奖励

当前正式实现只有 `atomic-entailment-v3-aligned-contradiction`，旧 lexical 和“lexical + NLI”路径已经删除。

## 为什么这样设计

BLEU/ROUGE 或手写同义词表会把解释变成关键词匹配；整段给一个 NLI 分数又无法区分“漏掉关键证据”和“额外编造证据”。当前实现利用 TASLE 已有的对象异常、异常开始、异常结束三类结构化标注，先形成样本自己的 evidence facts，再把生成解释切成 atomic claims。

冻结的 entailment cross-encoder 对每个 `fact → claim` 给出 entailment/contradiction 概率。随后按 entailment 做一对一最大权匹配：一个重复 claim 不能覆盖多个事实，一个事实也不能反复给多个 claim 加分。矛盾分只读取已经匹配的 claim/fact 对，不再让正确的结束描述与主要异常或开始阶段交叉比较。

```text
coverage = 带权事实召回率（对象异常权重 2，开始/结束各 1）
precision = 生成 claim 中有事实支持的比例
reward = 0.55 * coverage + 0.45 * precision
         - 0.50 * contradiction
```

奖励裁剪到 `[-1, 1]`。它只在正 proposal 的定位 IoU 达到门槛后开启；负 proposal 只应输出规范 no-event。正式实验与 SFT 一样使用人工解释标注，因此不传 `--require-candidate-observable`（默认关闭）；若以后补充逐样本可观察性审计，可显式加上该开关。

## 边界

这是冻结文本蕴含模型对人工证据标注的评分，不等同于重新观看视频验证每个 claim。它比词典匹配可审计、可复现，也适合在线评估一组 rollout；论文仍应在独立测试集上补充人工解释评测，并报告 coverage、claim precision、contradiction 和 joint localization-explanation success。

实现位置：

- `trace/text_explanation_reward.py`：事实、claim、一对一匹配和奖励；
- `trace/opd_grpo.py`：定位门控与三类 reward 组合；
- `trace/trace_trainer.py`：structure-aware token mask 与指标；
- `README_RUN_OPD_GRPO.md`：正式 Python 命令。

设计依据包括 SPICE 的结构化语义匹配、CIDEnt 的 entailment reward，以及 GKD/GRPO 对冻结评分器和在线 rollout 的使用方式。这里没有复制其代码，只复用公开方法的基本思想。
