# 面向 TASLE/MSLoc 的定位—解释联合 GRPO：从 OPD 到可验证解释

## 摘要

本方案将第二阶段训练分为 `ref2 SFT → RPE-OPD → structure-aware GRPO`。SFT 学会 Trace 的事件时间流与基础说明；OPD 用训练期的真实/AIGC 成对证据降低正 proposal 的过度拒绝；最后的 GRPO 只让部署学生看到 candidate 视频，在第一阶段真实产生的 proposal 上在线采样多条输出，用可验证的定位奖励和基于证据要点的解释奖励进行相对优化。关键判断是：定位可直接由 GT 时间段和解析器规则评分；解释不能用 BLEU/ROUGE 或单个 LLM 总分作为主奖励，而应拆为原子证据要点的覆盖、生成 claim 的视觉支持和时间一致性。解释分只在定位达到最低正确性时生效，避免“定位错但说得漂亮”获得奖励。

## 1. 研究问题

- RQ1：如何在 Trace 的专用 time-token 输出协议上用 GRPO 同时优化有效定位与解释，而不把格式错误误奖为 real？
- RQ2：如何把 TASLE 的 rationale 转成可被自动、稳定评分的解释奖励，而不奖励空泛或幻觉性文字？
- RQ3：OPD 与 GRPO 如何分工，避免训练期 paired reference 泄漏为测试期无法观察的解释？

## 2. 调研结论

GRPO 的原始机制是在同一个 prompt 上由当前/旧策略在线采样一组 completion，用组内奖励均值和方差得到相对 advantage，因而不需要价值网络；目标仍保留 PPO clipping 和到冻结 reference policy 的 KL 正则 [1]。因此本任务应以**同一 proposal 的多条学生独立输出**为一组，而不是使用教师答案或固定 SFT 答案。DeepSeek-R1 的经验还表明，格式和可验证正确性适合用规则奖励；自由的神经 reward model 易被过优化 [2]。

解释评测研究同时给出三条约束：第一，单条 gold rationale 的表面相似度会漏掉不同但有效的解释，不能将 BLEU/ROUGE 作为主奖励 [9]；第二，将文本拆成原子事实并分别验证支持性，比整段二元判断更合适 [5]；第三，视觉解释需要考察文本 claim 是否由画面支持，而不只是语言是否流畅 [6]。面向 deepfake reasoning 的近期工作也采用人工校准的 pointwise/pairwise 多模态 judge，而非词面指标 [7]。LLM judge 可作冻结 claim verifier，但其长度、文风偏好必须通过人工校准和对抗样本测试；它不能是唯一 reward。

对本任务尤为重要的是，TASLE 的原始 rationale 是在“真实参考—AIGC”对照下生成并人工修订的 [3]。其中一部分差异可能只在 reference 存在时可观察。故 GRPO 的**主解释 reward 与主解释 judge 都必须 candidate-only**；paired reference 只能作为训练期辅助消融，不能作为主分数。

## 3. 训练总流程

```text
Stage 0: ref2 SFT
  学生只看 candidate proposal，学习 Trace 的文本/time/score 输出协议。

Stage 1: RPE-OPD
  学生仍只看 candidate；冻结教师额外看逐帧对齐的 reference/candidate pair。
  只蒸馏事件分支、时间流和 candidate-observable 的短证据 token；
  guided rollout 仅用于部分困难正 proposal。

Stage 2: GRPO（本方案）
  actor 初始值 = Stage-1 OPD 学生；reference policy = 该 checkpoint 的冻结副本。
  教师 pair、guided rollout 均退出训练环；每个 proposal 只以 candidate 输入学生。
```

Stage 2 的目标不是继续让学生模仿 paired teacher，而是在实际部署分布上，在“准确定位”和“证据充分、可读的解释”之间做最终优化。

## 4. GRPO 数据与在线采样

### 4.1 训练单位

仍使用第一阶段在训练集实际产生的 proposal replay buffer，而非 oracle GT 窗口。每条记录至少保存：`video_id`、proposal `[p_s,p_e]`、与所有 GT 段的交集及相对时间、真假标签、`bnd_cot_st`、`obj_cot`、`bnd_cot_ed`、采样帧时间轴和数据来源。

正 proposal 是与任一 GT 有交集的 proposal；负 proposal 是无交集的 near-negative 或真实/第一阶段误报 proposal。每个 batch 应在难正、普通正、near-negative、纯真实之间分层，以免 GRPO 学成“永远报 fake”。

### 4.2 组采样

对同一个 proposal (x)，从旧策略 \(\pi_{old}\) 在线采样 (G) 条完整输出 \(y_1,\ldots,y_G\)，建议先以 (G=4\) 或 (8) 做小规模试验。采样使用适度温度以产生不同的边界和解释；所有 group member 都只输入同一 candidate 视频。

组内奖励无方差时没有相对学习信号。因此必须记录 `group_reward_std` 和 `all_same_rate`；若一组完全相同则跳过 actor 更新或重新采样，不能把除零当作零优势继续训练。

## 5. 先修复输出解析：奖励的唯一事实来源

每条 completion 必须保存 raw output IDs、解析出的所有时间段、caption、以及失败原因。当前协议中，正事件是非空时间流：两个固定格式时间值、`<sep>` 和 time-`<sync>`；真实目标是空时间流（唯一 time-`<sync>`）加 `No forgery.`。

奖励解析器必须区分：

| 情形 | 训练语义 |
|---|---|
| 合法双时间段 | valid event |
| 空时间流且规范 no-forgery 文本 | valid no-event |
| 声称伪造但时间不足、不可解析、反序或越界 | format failure |
| 空输出或混合结构 | invalid |

尤其不能把所有无有效 segment 的输出直接当作 real 并给奖励。否则模型可以故意生成坏时间流来逃避定位责任。

## 6. 奖励设计

所有基础奖励先限制在 \([-1,1]\) 或 \([0,1]\)，再组合；这样某个分量不会仅因尺度大而支配组内 advantage。

### 6.1 定位与真假奖励 \(R_{loc}\)

对正 proposal：

\[
R_{loc}^{+}=V\,[0.25\,D+0.55\,IoU_{match}+0.20\,B]-0.25\,N_{extra}.
\]

其中 (V) 是时间流是否合法；(D) 是是否报出至少一个事件；\(IoU_{match}\) 是预测段与 GT 相交段的最佳一对一匹配 IoU；(B) 是起止边界误差归一化分数；(N_{extra}) 惩罚无匹配的额外伪造段。无事件、格式失败或空输出不因被解析器默认归 real 而得分。

对负 proposal：

\[
R_{loc}^{-}=
\begin{cases}
+1,& \text{合法空时间流且明确 no-event}\\
-1,& \text{生成任一合法 fake event}\\
-0.5,& \text{格式错误或语义含糊。}
\end{cases}
\]

这使 GRPO 同时抑制正样本过度拒绝和负样本 false event。

### 6.2 格式奖励 \(R_{fmt}\)

使用小权重、完全确定的规则奖励：合法事件结构或合法 no-event 模板为正，非法时间、超窗口时间、无结束时间、过多未匹配 segment 为负。它只帮助保持 Trace 协议，不能替代定位 reward；若把格式权重设太大，模型会只学会输出漂亮的时间 token。

### 6.3 要点化解释奖励 \(R_{exp}\)

先离线将现有人工筛查过的 annotation 变为 `evidence card`，而不是直接拿整段 caption 做相似度：

```json
{
  "manipulation_type": "temporal | spatio-temporal",
  "start_boundary": {"class": "...", "candidate_observable": true},
  "entity_or_region": {"class": "...", "candidate_observable": true},
  "visible_change": ["..."],
  "end_boundary": {"class": "...", "candidate_observable": true},
  "pair_only_claims": ["..."]
}
```

现有 annotation 已天然提供该结构：普通 temporal 样本可使用 `bnd_cot_st.bnd_class`、`obj_cot.bnd_sub_class`、`bnd_cot_ed.bnd_class` 三类要点；spatio-temporal 样本至少使用 `obj_cot.bnd_sub_class`。`pair_only_claims` 不进入主 reward。

将生成 caption 切分为 atomic claims，并用冻结、独立的 candidate-only verifier 对每个 claim 判定 `supported / contradicted / unverifiable`，同时判断 evidence card 的每个可观察要点是否被覆盖。定义：

\[
R_{exp}=\mathbb{1}[V\land IoU_{match}\ge \tau]
\left(0.45\,C+0.35\,P+0.20\,T\right)-\lambda_h H.
\]

- (C)：加权要点覆盖率；必须同时提到实体/区域与异常/变化，泛泛的“存在不一致”不计覆盖；
- (P)：生成 claim 中被 candidate 视频支持的比例；
- (T)：claim 所述变化与预测时间窗口相符；
- (H)：矛盾或明显幻觉 claim 数；
- 门槛 \(\tau\) 令定位至少达到预注册 IoU 下限后解释分才开启。

简洁性采用硬约束（例如单个短 caption、最大 token 数）和超长轻罚，不按华丽程度给正奖励。负 proposal 不要求编造解释；它只奖励规范的 no-event，任何虚构 artifact 均受罚。

### 6.4 verifier 的准入条件

先从训练集外的验证子集抽取专家双标/多标样本，人工分别评估：要点覆盖、claim 视觉支持、时间一致性、简洁有用性。自动 verifier 只有在与盲评人工存在足够相关性、且能通过下列攻击测试时才可进入 RL reward：

- 将 evidence card 随机错配到别的视频；
- 把正确 claim 的实体、属性或时间反转；
- 只罗列高频 artifact 词；
- 用冗长、流畅但无视觉证据的文本；
- 删除或错位候选视频帧。

若未通过，解释 scorer 仅作为评测工具，不进入 GRPO；GRPO 先只优化定位和格式。

## 7. GRPO 目标：基础版与结构感知版

### 7.1 可复现的基础 GRPO baseline

每个输出的总 reward 为：

\[
R_i=w_lR_{loc,i}+w_fR_{fmt,i}+w_eR_{exp,i}.
\]

在同一 proposal 的 group 内标准化：

\[
A_i=\frac{R_i-\mu(R_{1:G})}{\sigma(R_{1:G})+\epsilon}.
\]

再以原始 GRPO 的 clipped likelihood-ratio objective 更新，并以冻结 OPD checkpoint \(\pi_{ref}\) 做 KL 正则：

\[
\mathcal L_{GRPO}=-\frac{1}{G}\sum_{i,t}
\min(\rho_{it}A_i,\operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)A_i)
+\beta\,D_{KL}(\pi_\theta\Vert\pi_{ref}).
\]

这是与原始 GRPO 最接近、必须首先报告的 baseline。

### 7.2 推荐主方法：结构感知 masked GRPO

原始 outcome GRPO 会把同一个序列奖励广播给所有 token：高定位分也会奖励无关或错误的长解释。针对 Trace 的交织输出，推荐将 reward advantage 分开并加在对应 token 区域：

- (A_{loc})：仅作用于 text-`<sync>`、time slots、数字、`<sep>`、time-`<sync>`；
- (A_{fmt})：仅作用于结构 token；
- (A_{exp})：仅作用于 caption 文本 token；
- `<score>` 当前无信息，mask 为零。

\[
\mathcal L_{S\text{-}GRPO}=-\frac1G\sum_{i,t,k\in\{loc,fmt,exp\}}
M_{it}^{k}\min(\rho_{it}A_i^k,
\operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)A_i^k)
+\beta D_{KL}(\pi_\theta\Vert\pi_{ref}).
\]

它是对 GRPO 的任务特定扩展，论文中必须明确称为 `structure-aware/masked GRPO`，不可称为原始 vanilla GRPO。其必要性应由与 7.1 的对比消融证明。

## 8. 稳定训练设置

- actor 初始值与 KL reference 都取 OPD 最终学生 checkpoint；GRPO teacher 不是 paired teacher；
- Stage 2 不使用 guided rollout，否则采样不再代表部署学生策略；
- 先运行定位/格式 GRPO 小阶段，说明 parser、F1Loc、false-event rate 稳定后才渐增 (w_e)；
- 保持小学习率、PPO clip 和自适应/监控 KL；记录 reward 均值、方差、各分量、KL、clip fraction、输出长度、有效时间流率；
- 可保留少量普通 ref2 SFT replay microbatch 作为辅助锚定：
  \(\mathcal L=\mathcal L_{GRPO}+\eta\mathcal L_{Trace-SFT}\)。这是工程防退化措施，不是原始 GRPO；
- 若负 proposal false-event rate、格式失败率或 OOD 指标恶化，降低解释 reward 权重、提高 KL/辅助 SFT，并回查 reward 攻击。

## 9. 必须报告的验证与消融

### 9.1 主结果

- F1Det、F1Loc、边界 MAE、positive-proposal ORR（拆 semantic/format）、negative false-event rate；
- `valid-event rate` 与 raw parser 失败类型；
- 解释的要点 coverage、claim precision、时间一致性、幻觉率；
- `joint success = 定位正确 ∧ 解释达到阈值`。同时报告条件解释质量（只在定位正确时）和联合指标，避免其中一项掩盖另一项。

### 9.2 人工评测与 reward 审计

在未参与 reward 构造的 held-out、unseen/OOD/tool 分层上，进行盲评 pointwise 与 pairwise 解释评测。报告自动 (R_{exp}) 与人工评分的相关性/一致性，并保留错配 card、通用词堆砌、长文本和 claim 反转攻击的结果。

### 9.3 消融矩阵

1. ref2 SFT；
2. SFT + OPD；
3. OPD + vanilla GRPO（仅定位/格式）；
4. OPD + vanilla GRPO（加入解释 reward）；
5. OPD + structure-aware GRPO；
6. 去除解释定位门控；
7. 去除 claim precision/幻觉罚；
8. candidate-only judge 与 pair-aware judge 对比；
9. 无 KL、无辅助 SFT 的稳定性对比。

若解释自动分数上升但人工盲评、joint success 或 OOD 没有提升，即视为 reward hacking，不主张解释能力提升。

## 10. 对 OPD 设计的必要修改

1. OPD 中不能蒸馏整段 caption。paired teacher 能看到而 candidate 学生看不到的差异，可能被写成部署时不可证实的解释。OPD caption mask 最多保留 `candidate-observable`、与人工要点一致的短证据 span；时间/事件 token 仍是主蒸馏对象。
2. 在 OPD 前先实现 raw token 与拒绝原因日志；GRPO 完全依赖该解析器，不能继续把所有 invalid output 静默归 real。
3. OPD 结束后冻结学生作为 GRPO reference；Stage 2 不再给 actor 或 reward 主路径输入 paired reference，也不再 guided。
4. 先构建并人工审计 evidence card；这是解释 GRPO 的前置条件，而非可在 reward 不可信时跳过的细节。

## 11. 结论

这条训练链的分工是清晰的：SFT 学会协议和基本解释，OPD 用训练期成对证据解决“正 proposal 却不报事件/时间”的支持不足，GRPO 则在真实 proposal 和真实学生输出分布上选择更准确的边界与更可核验的说明。方法的贡献不应表述为“用 GRPO 让文字更像标注”，而应表述为：在不改变部署输入的前提下，以定位门控的原子证据奖励，将时间定位与 candidate-grounded explanation 联合优化；并以 human-calibrated reward audit 证明改进不是格式、长度或关键词投机。

## 参考文献

[1] Z. Shao, P. Wang, Q. Zhu, et al., “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models,” arXiv:2402.03300, 2024.

[2] D. Guo, D. Yang, H. Zhang, et al., “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning,” arXiv:2501.12948, 2025.

[3] Y. Feng, J. Li, Q. Lu, et al., “Explainable Forensics of Manipulated Segments in Untrimmed Long Videos,” arXiv:2606.02402, 2026.

[4] Y. Liu, D. Iter, Y. Xu, et al., “G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment,” EMNLP, 2023.

[5] S. Min, K. Krishna, X. Lyu, et al., “FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation,” EMNLP, 2023.

[6] L. A. Hendricks, R. Hu, T. Darrell, and Z. Akata, “Grounding Visual Explanations,” ECCV, 2018.

[7] K. Kuckreja, P. Gupta, M. H. Khan, and A. Dhall, “Pixels Don’t Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision,” CVPR, 2026.

[8] Y. Guo, J. Liu, M. Li, et al., “TRACE: Temporal Grounding Video LLM via Causal Event Modeling,” ICLR, 2025.

[9] P. Jansen, K. J. Smith, D. Moreno, and H. Ortiz, “On the Challenges of Evaluating Compositional Explanations in Multi-Hop Inference: Relevance, Completeness, and Expert Ratings,” EMNLP, 2021.
