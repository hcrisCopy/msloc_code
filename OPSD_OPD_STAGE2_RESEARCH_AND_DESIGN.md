# 面向 MSLoc 第二阶段过度拒绝的成对证据 OPD 设计

## 结论先行

推荐把第二阶段做成 **RPE-OPD（Reference-Pair Evidence-guided On-Policy Distillation）**：部署时的 Trace 学生仍只看第一阶段给出的 proposal；训练时，冻结的教师额外看到与该 proposal 时间对齐的“原始真实参考段（上）/被替换的 AIGC 段（下）”竖向拼接视频。学生在自己的实际生成轨迹上得到教师的 token-level 蒸馏，而不是只模仿一份固定答案。

不过不建议直接把普通 OPSD 的全序列 KL 搬过来。当前问题恰好是“正候选被拒绝、没有生成时间 token”，这是一个罕见而结构化的行为。应采用 **教师引导的正候选 rollout + 证据掩码 + 时间 token 加权**；同时保留真实/误报 proposal，避免模型学习成“第一阶段一报就必定 fake”。

这个设计的主张应当是：*训练期可用的逐帧真实参考证据，使 Trace 在没有参考视频的测试期仍能恢复候选中的伪造事件及其边界。* 它不是给测试样本偷看 reference，也不是把 GT 时间戳作为教师输入。

---

## 1. 先把当前失败定位清楚

### 1.1 原论文的标注流程与当前 Trace 是两回事

论文 TASLE 的 rationale pipeline 对每一段 AIGC 内容提供：

1. 被替换前的真实参考片段；object-level 情况为 pre-masked 真实视频；
2. 对应的 AIGC 片段；
3. 用于起止过渡解释的额外 boundary-context 视频；
4. 将参考段和生成段**竖向拼接**，让 Qwen3-VL-235B 可以逐帧比较；其生成 rationale 最后由六位人工标注者筛查和修订。

这份论文说明的是高质量 rationale 的生成/标注过程，而非 MSLoc-PR 的输入。MSLoc-PR 推理只处理原始长视频内的 candidate proposal；它没有看到原参考片段。因此，竖向视频对正是一个合法、很强的**仅训练期特权信息**，也正好对准论文中已被证明有用的比较证据。

### 1.2 当前仓库的 `ref2` 行为

当前实现并非把完整视频送入第二阶段：

| 环节 | 当前行为 | 对过度拒绝的影响 |
|---|---|---|
| 训练样本 | 读取第一阶段 JSON 的每个 `model_inference.segment`，每个 proposal 建一个独立样本 | 已经是 proposal-conditioned，而不是 oracle full-video 训练 |
| 正负标签 | 与任一 GT segment **有任意正交集**即作为 fake，并仅监督 proposal 窗口内的 GT 交集；无交集则监督 `No forgery.` | 视觉稍弱的正 proposal 与真实 false proposal 都会出现，模型会学习“候选不一定是假” |
| 采样 | proposal 左/右各 `bnd_ratio=0.2` 区域密采样 16 帧，中间稀采样 8 帧 | 共 40 帧，的确是 boundary-aware，但没有参考对比 |
| 生成/解析 | prompt 后强行接 `<sync>`；只有解析到至少两个 time token 才保留预测 | 模型生成了“真实/无伪造”、空文本或不合法时间串，都会被记作 `[-99,-99]` 并最终归为 real |

对应代码证据：proposal 读取在 `Trace/trace/train_mt.py:732-771`，正交集判定在 `:1044-1059`，负 proposal 的目标在 `:1117`；采样函数在 `Trace/trace/mm_utils.py:558-698`；推理端在 `Trace/trace/eval/evaluate_ref.py:352-533`。所以目前测到的“过度拒绝率”包含两种不可混为一谈的错误：

* **语义拒绝**：模型认为该候选没有伪造，因而不发出 event/time token；
* **格式拒绝**：模型本来试图回答，但 time-token 序列不足两个、无效或被解码器丢掉。

在做 OPD 前，评测必须记录原始 token 序列，以分别报告这两类错误；否则不可能知道改进来自鉴伪能力，还是仅来自格式服从。

另有一个应先清理的**易混淆遗留参数**：`ref2.sh` 设置 `--num_frames 32`，但 ref2 路径直接调用 `process_video_ref_split(bnd_frames=16, seg_frames=8)`，实际返回 `16+8+16=40` 帧。`--num_frames` 属于 `DataArguments`，在 ref2 这条取样路径不决定帧数；而当前 video encoder 使用输入 tensor 的实际时间长度。故这不是已证实的 32/40 位置编码错误，但会误导实验记录、模型配置和结果文件命名。OPD 脚本应删除该歧义或明确记录 `sampled_frames=40`；若改用 `ref_projector`，其硬编码切分也要求 40 帧。

---

## 2. OPD/OPSD 调研：哪些思想可直接借鉴

以下结论以论文原文/官方会议页为主；2026 工作目前多数为预印本，不能把它们当作已完成同行评审的定论。

| 工作 | 已验证的核心思想 | 对本问题的可借鉴点 | 不应照搬之处 |
|---|---|---|---|
| [GKD, ICLR 2024](https://arxiv.org/abs/2306.13649) | 用学生自己生成的序列来做 token-level teacher feedback，避免固定 teacher demo 与部署轨迹不匹配 | 过度拒绝正是学生自己会走到的坏轨迹；必须在这些轨迹上教，而不只是 SFT GT rationale | 普通 GKD 不含特权视觉证据 |
| [Self-Distilled Reasoner / OPSD](https://arxiv.org/abs/2601.18734) | 同一模型因上下文不同而成为 PI-teacher / student，并在 student rollout 上最小化每 token 分布差异 | 可用同一个 Trace checkpoint 的双条件策略，降低额外教师依赖 | 其 PI 是数学解答，不能假定所有 token 都适合视频取证蒸馏 |
| [Privileged Information Distillation](https://arxiv.org/abs/2602.04942) | 研究 PI-teacher 到无 PI policy 的转移，并给出 OPSD/反向 KL 变体 | 明确支持“训练时可见、部署时不可见”的设定 | 任务是 agent/action，视频编码和结构化时间输出需另行实现 |
| [Vision-OPD](https://arxiv.org/abs/2605.18740) | crop-conditioned teacher 把局部证据的感知能力蒸馏给 full-image student | 最接近“reference pair 让伪造差异更显著”的区域/全局感知鸿沟 | 它不处理双边界及 TRACE 特殊时间 token |
| [GUI-SD](https://arxiv.org/abs/2605.00642) | 以 box + Gaussian mask 形成不直接泄露坐标的视觉 PI，并按数字重要性/教师置信度加权 | 起止时间的数字 token 是最该加权的位置；教师可只看视觉对，不看 GT 数字 | 不能把 GT box/mask/time 直接塞进教师 prompt，否则会变成坐标复制 |
| [EDGE-OPD](https://arxiv.org/abs/2605.23493) | 稀有目标行为可能不会出现在学生 rollout；用 guided rollout 注入它，并只更新 PI 支持的 evidence token | 对“学生根本不发 timestamp”最关键：需引导正候选进入 event/time 分支，再只蒸馏 event、时间、证据 token | 不要全输出蒸馏，否则会蒸馏长度、文风等副作用 |
| [ViGOS](https://arxiv.org/abs/2606.19120) | MLLM OPSD 会走视觉 shortcut；把感知与推理解耦，对无效轨迹用 reference teacher 恢复格式 | 应把“观察到的边界/差异”与最终 timestamp/rationale 分层；格式无效轨迹单独恢复 | 不要让教师只依据隐藏在 prompt 中的 textual label 给答案 |
| [Clue-OPSD](https://arxiv.org/abs/2608.25356) | 长视频 student 从看 clue interval 的 teacher 学习，测试时不需要 clue | 与 proposal + boundary evidence 的时序结构高度相近，支持用训练期视觉线索做 PI | 本工作是长视频问答，不含 reference-pair forgery comparison |
| [ViTED, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Lu_VITED_Video_Temporal_Evidence_Distillation_CVPR_2025_paper.html) | 先构造并搜索时间证据链，再蒸馏模型产生证据区间 | 可把“候选内的 r2f、内部、f2r 证据”显式化，并单测 evidence localization | 它是 VideoQA，不解决正候选被拒绝 |

两组重要的反例约束了方法叙事。`EDGE-OPD` 指出特权上下文也会改变无关 token（长度、风格、局部偏好），故需要 evidence mask。两篇近期分析，[OP²SD](https://arxiv.org/abs/2608.09228) 和 [Rethinking PI in OPSD](https://arxiv.org/abs/2608.18271)，都发现“正确 reference”有时并不比其他上下文更有利。故论文实验必须包含随机错配 reference、时间错位 reference 和普通 OPD 的控制组，才能证明收益真的来自**成对视觉证据**而不是教师被额外上下文改变后的说话方式。

---

## 3. 方法：RPE-OPD

### 3.1 两个条件策略

对第一阶段 proposal (P=[p_s,p_e])，按当前 ref2 的 16/8/16 策略取视频 (V_P)，并保留相对时间轴 (\tau_P)。

\[
\pi_S(y\mid x_S)=\pi_\theta(y\mid Q,V_P,\tau_P),\qquad
\pi_T(y\mid x_T)=\pi_{\bar\theta}(y\mid Q,V_P,\tau_P,C_P).
\]

`Q` 为现有 Trace forensic prompt，学生输入 (x_S) 与现有部署完全一致。(C_P) 是**只给教师**的成对证据：将真实 reference 帧固定放上方、candidate AIGC 帧固定放下方，按原始时间索引一一对齐，逐帧组成一个 composite video。教师 prompt 明确说明“上方为替换前真实参考，下方为待检候选；请仅依据可见差异判断”。

不把真值时间、`fake/real` 标签、人工 rationale 文本或 object mask 作为基本教师输入。这样教师仍须视觉判断，学生才可能内化视觉线索。MV2V 的 mask 可作为单独的更强 PI 消融，而不能混入主方法。

原论文的 boundary context 不应被误解为要在 proposal 外再凭空找 reference：学生视频 (V_P) 已含 proposal 两端上下文；教师的 paired reference 主要覆盖被替换的时间段，边界视觉仍从 (V_P) 的两端获取。若显存允许，教师还可增加一条短的 “paired transition strip”，但必须保持学生没有这条输入。

### 3.2 proposal replay：先让训练分布等于真实失败分布

每轮 OPD 以当前 DINOv3/DeMamba 第一阶段在训练集的预测 JSON 建立 replay buffer，而不是只使用 GT segment。每个 proposal 存储：视频 ID、proposal、所有 GT 交集、来源 stage-1 score/class、是否曾被 stage-2 拒绝、reference 对的可用性。

建议采样配比起点为：

* 35%：正 proposal，完整覆盖 GT；
* 35%：困难正 proposal，边界偏移、只覆盖一侧、窗口过宽/过窄、或当前 Trace 输出为空；
* 20%：近邻 hard negative，紧贴 GT 但无交集；
* 10%：纯 real/第一阶段 false positive。

正样本的目标区间是 proposal 与 GT 的交集后再转换为相对时间，而不是把原始 GT 原封不动放入窗口。若一个 proposal 与多个 GT 相交，保持多个 event；不要无提示地只留第一个。保存 `positive_overlap`、交集时长和 IoU，后续可分层报告。

reference pair 在 TASLE 训练集的构造文件中应由“生成前源片段/生成后片段”的时间索引恢复。开始开发前需要检查资产是否确实保存了这层映射；当前 `ref2` JSON 只使用视频路径、proposal、`annotations`，代码本身未读取 reference 视频路径。若没有保存，应从 TASLE 生成流水线的 `combine_dir`/源片段记录重建映射；不能拿同一段篡改后视频复制两次伪造 `C_P`。

### 3.3 teacher warm-up 与 on-policy 轨迹

1. **学生 SFT warm-up**：现有 `ref2` 训练，修正 40 帧配置，并保留正/负 proposal。
2. **特权教师 warm-up**：从同一 Trace 初始化，以 composite video 替换/补充教师视觉输入做短暂 SFT；只保留教师解码能通过 GT 验证的样本。得到冻结快照 (\bar\theta)。
3. **学生 on-policy rollout**：学生对 (x_S) 采样输出 (y\)，并保存 token、logprob、原始文本以及解析状态。采样温度可从 0.7 起，不能只用 greedy，否则很少看到拒绝分支附近的替代 token。
4. **教师评分同一前缀**：教师只在 `no_grad` 下前向，输入 (x_T) 但强制喂入同一个学生前缀 (y_{<t})，得到每一位置的 (p_T(\cdot\mid x_T,y_{<t}))。这一步而不是教师另生成一篇答案，才是 OPD。
5. 每个 epoch 或固定 K step 更新一次教师快照/EMA，K 内固定；不要让梯度通过教师。教师与学生可共享 Trace 基座、不同 adapter/head snapshot；若使用完整副本，教师无梯度但会增加一次前向计算。

### 3.4 处理“没有 timestamp”的 guided rollout

普通 OPSD 有一个鸡生蛋问题：正候选上学生若几乎总走向 `No forgery`/空输出，则正 event/time token 很少出现在 on-policy data，蒸馏信号也到不了这些位置。对此仅对**正 proposal**的部分 rollout 使用：

\[
q_t=\alpha p_T(\cdot\mid x_T,y_{<t})+(1-\alpha)p_S(\cdot\mid x_S,y_{<t}),
\]

并从 (q_t) 采样。初始 (\alpha=0.5\)，只在第一个 event-time token、两个边界 time token 及其附近的前缀开启；随后线性退到 0。剩余 rollout 仍是纯学生采样。对于学生生成空 event、缺失第二时间戳或解析失败的**正样本**，强制至少采一条 guided trajectory；这是借鉴 EDGE-OPD 的必要改造。

负 proposal 不做“伪造方向”的引导，仍让学生自己输出 no-event，从而保存拒绝真实候选的能力。

### 3.5 损失与 evidence mask

对学生轨迹使用：

\[
\mathcal L = \mathcal L_{Trace\text{-}SFT}
+\lambda_{opd}\frac{1}{\sum_t m_t}\sum_t m_t w_t
\operatorname{JSD}\big(p_S^t,p_T^t\big)
+\lambda_{anchor}\operatorname{KL}(p_S\Vert p_{S,0}).
\]

* `L_Trace-SFT`：保留原有 `<sync> / <time> / <score> / caption` 的监督，作为格式锚点；
* (m_t)：只选 event 是否存在的分叉、`<sync>`/`<time>`/分隔符、两端 timestamp 数字、与 GT rationale 或高置信教师一致的短证据词；普通礼貌语、长风格性 rationale 和无关 explanation 置零；
* (w_t)：时间数字和 event 开关权重大；再乘教师置信度，例如 (1-H(p_T)/\log |\mathcal V|)。这对应 GUI-SD 的“重要坐标/高置信 token 优先”；
* 仅在教师对该正 proposal 的独立贪心解码能通过验证时启用 OPD：输出合法、至少两个 timestamp、与交集 GT 的 IoU 达门限、caption 不明显矛盾。未通过时仍保留 SFT，不能把错误特权教师蒸馏给学生；
* (p_{S,0}) 是 warm-up 学生或 base Trace，仅小权重锚定，防止把一般 Trace 行为冲坏。

实现初版建议选 JSD 或 teacher-to-student forward KL，temperature 1--2；不要一开始叠加 PHF 一类隐藏层 loss。Trace 的视觉/时间交织 token 与双模态 projector 已足够复杂，先证明 token-level evidence OPD 有效，再研究隐藏流蒸馏才可解释。

### 3.6 输出协议：修复观测，避免“伪改善”

主实验应保持 TRACE 的 event token 格式，以便与原 MSLoc 公平对比；不要把 JSON schema 改动混入方法增益。与此同时应做两件事：

1. 评测器保存 raw generated token IDs 与解析失败原因（无 `<sync>`、不足 2 个 time token、非有限数、时间反序、越界），然后报告 format failure；
2. 一个独立消融可改成显式 `NO_EVENT`/`EVENT` decision token，要求每个 proposal 都产生决定，再由 event 分支给时间。它有工程价值，但由于改变了解码协议，必须与主结果分开报。

---

## 4. 实施落点（尚未改代码）

最小侵入路径是让教师也只接收**一条视频**：把每个对齐帧上下拼成一个 RGB 帧，故不必改 Trace 的单视频接口，也不改变 student 输入、时间 tokenizer、DAM/EAM 或部署脚本。真正需要新增的是双视图 batch 和自定义 loss。

| 文件/组件 | 修改内容 |
|---|---|
| 新建 `Trace/trace/opd_replay.py` | 读取 stage-1 prediction JSON + GT；产出正、困难正、近邻负、真实负 proposal replay，以及 GT 相对区间 |
| 新建 `Trace/trace/reference_pair.py` | 从训练期 source/generated 映射加载对齐帧，固定 `top=reference, bottom=candidate`，生成教师 composite；严格检查 fps、帧数、时间戳和缺失映射 |
| `Trace/trace/train_mt.py` | 新增 `train_mode=opd_ref2`、`reference_index_path`、`teacher_checkpoint`、`opd_*` 参数；dataset/collator 同时返回 `video_student`、`video_teacher`、两个时间轴和 token mask |
| `Trace/trace/trace_trainer.py` | 覆写 `compute_loss`：无梯度教师前向取 logits、学生 rollout 前缀对齐、计算 masked/weighted JSD + 原 Trace loss；记录 `opd_loss`、teacher-valid rate、guided rollout rate |
| 新脚本 `Trace/scripts/train/opd_ref2.sh` | 先 warm-up teacher，再训练 student；明确实际采样为 `--bnd_frames 16 --seg_frames 8`（40 帧），删除或注释不参与 ref2 采样的 `--num_frames 32`，并传入 replay/reference index |
| `Trace/trace/eval/evaluate_ref.py` | 保留原 parser，同时保存 raw output 和 `reject_reason`；增加 positive-proposal over-rejection 指标，绝不在解析失败时悄悄删除诊断信息 |

分布式实现上，教师不应参与 optimizer 或反向传播。完整教师副本在 ZeRO-3 下会增加显存/通信复杂度；若八卡显存不够，先固定 backbone，只为教师保留冻结 adapter/head snapshot，或把教师前向放到独立 rank/group。不要以“同一个可训练模型在同一次 forward 中既当 teacher 又回传梯度”替代冻结 teacher，否则目标会随更新漂移且很难复现。

---

## 5. 必须报告的实验，而不只是 F1Det/F1Loc

### 5.1 新的诊断指标

在 GT 有交集的 stage-1 proposal 集合 (\mathcal P^+\) 上定义：

\[
\mathrm{ORR}=\frac{\#\{P\in\mathcal P^+:\text{输出空 event、NO\_EVENT 或解析不到合法双时间戳}\}}{|\mathcal P^+|}.
\]

同时报告：

* `semantic-ORR`：原始回答明确否定/`No forgery`；
* `format-ORR`：尝试 event 但语法或时间非法；
* `positive proposal recall`、`F1Loc`、`F1Det` 与原论文 rationale quality；
* `negative proposal false-event rate`：真实/无交集候选被硬报 fake 的比例；
* `valid-event rate`、边界 MAE（r2f / f2r 分开）、短/长 segment、不同 tool、seen/unseen/OOD 分层；
* oracle proposal 与真实 stage-1 proposal 两种设置。前者测 refinement，后者测完整系统。

只有 ORR 降低且 negative false-event rate 未失控，才能称为“抑制过度拒绝”，而不是“让模型永远报 fake”。

### 5.2 对照与消融矩阵

1. 现有 `ref2` SFT；
2. offline privileged SFT（教师看 pair，但只拟合固定 GT）——分离“多看 reference”与 on-policy 的贡献；
3. 无特权普通 OPD；
4. pair-OPSD 全 token KL —— 验证 evidence mask 的必要性；
5. RPE-OPD（mask，未引导）；
6. 完整 RPE-OPD（mask + guided rollout）；
7. 真实对齐 pair 替换成随机其他视频的 reference；
8. 正确 reference 但时间错位；
9. object-level 加 mask 的扩展（单列，不作为主方法）。

第 7/8 组尤其重要：若它们也同样有效，论文应如实把贡献表述为“privileged-context induced behavior”，而不能声称学生学会了真实/生成的对比证据。

---

## 6. 建议的推进顺序与风险闸门

1. **先做只读审计**：检查训练资产是否保存 reference/generated 的精确映射；统计当前正 proposal 的 semantic-ORR、format-ORR、IoU 和长度分布；清理/记录 ref2 的 32 参数与实际 40 帧采样之间的歧义。
2. **建立 pair teacher SFT sanity check**：教师比学生在相同正 proposal 上显著降低 ORR，且时间 IoU 上升；若没有，这个 PI 本身无效，勿进入 OPD。
3. **跑小规模 replay OPD**：只使用高质量 teacher-verified 正样本和保留的 negatives，确认 guided rollout 能让 time token 出现，且 false-event rate 没有增加。
4. **全量训练与严格控制组**：最终才运行 seen/unseen/OOD 和 tool 分层实验。

停止条件也应预先写明：若正确 pair 并未超过时间错位/错配 pair，或者 ORR 降低完全由 format-ORR 降低构成，则不能把方法解释为“reference comparison improves visual forensics”；此时更适合把贡献定位为结构化解码/格式蒸馏，或回到提升 teacher 的视觉分辨能力。

## 参考资料

* [TRACE: Temporal Grounding Video LLM via Causal Event Modeling, ICLR 2025](https://arxiv.org/abs/2410.05643)
* [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes, ICLR 2024](https://arxiv.org/abs/2306.13649)
* [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734)
* [Privileged Information Distillation for Language Models](https://arxiv.org/abs/2602.04942)
* [GUI-SD: Learn where to Click from Yourself](https://arxiv.org/abs/2605.00642)
* [Vision-OPD: Learning to See Fine Details for Multimodal LLMs](https://arxiv.org/abs/2605.18740)
* [EDGE-OPD: Internalizing Privileged Context with Evidence Guided On-Policy Distillation](https://arxiv.org/abs/2605.23493)
* [ViGOS: Seeing Before Reasoning](https://arxiv.org/abs/2606.19120)
* [Clue-OPSD: Where to Look Matters for Long-Video Understanding](https://arxiv.org/abs/2608.25356)
* [ViTED: Video Temporal Evidence Distillation, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Lu_VITED_Video_Temporal_Evidence_Distillation_CVPR_2025_paper.html)
* [OP²SD analysis of context-induced teacher behavior](https://arxiv.org/abs/2608.09228) and [Rethinking Privileged Information in OPSD](https://arxiv.org/abs/2608.18271)
