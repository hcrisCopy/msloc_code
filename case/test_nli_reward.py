# -*- coding: utf-8 -*-
"""
NLI 文字解释奖励 —— 小型演示脚本（看懂 atomic-entailment-v3 到底怎么打分）

背景
----
TRACE 第二阶段（GRPO）里，模型除了输出"定位"(timestamps)，还要输出"解释"(caption)。
解释奖励不看关键词、不用手写同义词表，而是用**冻结的 NLI 交叉编码器**，
把"模型生成的解释"和"TASLE 标注里的证据事实"做**文本蕴含**判断来打分。

一句话总结打分逻辑：
    1. 把标注切成若干"证据事实"(evidence facts)：对象异常 / 异常开始 / 异常结束
    2. 把模型解释切成若干"原子 claim"（按 . ! ? ; 或 while/whereas/however 切分）
    3. 对每个 (fact, claim) 组合跑 NLI，得到 entailment / contradiction 概率
    4. 按 entailment 做"一对一最大权匹配"（一个 claim 不能吃多个事实，一个事实不能被多个 claim 重复得分）
    5. 由匹配结果算 recall / precision，矛盾只罚"已匹配"的 claim/fact 对
    6. reward = 0.55*recall + 0.45*precision - 0.50*contradiction

本脚本做三件事：
    A. 从同级目录 MSLoc_data 的测试标注里挑 3 个样例（2 个 Round3 + 1 个 Round4）
    B. 给每个样例模拟 3 种模型解释：好 / 泛泛 / 反着说
    C. 详细打印每对 (fact, claim) 的 NLI 概率、一对一匹配、以及最终各项得分

结果同时写进 MSLoc_data/Trace/output/nli_case_demo/（markdown 报告 + json 原始数据）。

目录结构要求（脚本靠这个自动定位路径，不写死盘符）：
    <某个父目录>/
    ├── msloc_code/
    │   ├── case/test_nli_reward.py      <- 本脚本
    │   └── Trace/trace/...              <- 复用正式打分实现
    └── MSLoc_data/
        ├── data/Tasle-CoT-10K/annos/test_all_1209_0119.json
        └── Trace/ckpts/nli-deberta-v3-small   <- 冻结 NLI 模型

运行方式：
    conda activate vad            # 或任何装了 torch + transformers + sentencepiece 的环境
    cd <父目录>/msloc_code
    python case/test_nli_reward.py

    路径对不上的话，用命令行覆盖：
    python case/test_nli_reward.py \
        --nli-model /path/to/nli-deberta-v3-small \
        --annotation /path/to/test_all_1209_0119.json \
        --trace-dir /path/to/msloc_code/Trace \
        --out-dir /path/to/output \
        --device cpu            # 无 GPU 时
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

# ---------------------------------------------------------------------------
# 1. 默认路径 —— 全部相对脚本自身位置推导，可用命令行覆盖
# ---------------------------------------------------------------------------
CASE_DIR = pathlib.Path(__file__).resolve().parent            # msloc_code/case
CODE_ROOT = CASE_DIR.parent                                   # msloc_code
DATA_ROOT = CODE_ROOT.parent / "MSLoc_data"                   # 同级目录 MSLoc_data

DEFAULT_TRACE_DIR = CODE_ROOT / "Trace"
DEFAULT_NLI_MODEL = DATA_ROOT / "Trace" / "ckpts" / "nli-deberta-v3-small"
DEFAULT_ANNOTATION = DATA_ROOT / "data" / "Tasle-CoT-10K" / "annos" / "test_all_1209_0119.json"
DEFAULT_OUT_DIR = DATA_ROOT / "Trace" / "output" / "nli_case_demo"


def _load_module(name: str, path: pathlib.Path):
    """用 importlib 直接加载仓库里的正式源码模块，避免污染 Trace 包。

    好处：跑的就是线上真正用于 GRPO 的那份打分代码，不是抄写的副本。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到 Trace 源码：{path}\n请用 --trace-dir 指定 msloc_code/Trace 目录。")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 2. 样例定义：视频路径 + 展示名 + 三种模拟解释，三者绑在一起，不会错位
#    - 2 个 Round3：对象异常 + 异常开始 + 异常结束，三证据齐全
#    - 1 个 Round4：时空伪造，只有对象异常证据（展示"只有 object fact"分支）
# ---------------------------------------------------------------------------
SAMPLES = [
    {
        "video_path": "TVSum/Round3/videos/stitched/test/gzDbaEs1Rlg-20_57-96_06.mp4",
        "name": "手按中控台(gzDbaEs1Rlg)",
        "captions": {
            # 好：三条 claim 分别命中 对象/开始/结束 三个事实
            "good": (
                "The hand's fingers melt into the car's center console during pressing, "
                "showing unnatural articulation and inconsistent pressure. "
                "The hand movement becomes mechanically uniform and rigid without natural acceleration cues. "
                "At the transition, the man's head and eyes abruptly reorient with fluid motion after a stiff, static expression."
            ),
            "generic": "The video is fake.",
            # 反着说：与证据矛盾，应被 contradiction 狠狠扣分
            "contradict": (
                "The hand moves with natural finger articulation and consistent pressure. "
                "The hand's grip remains fluid and variable throughout. "
                "The man's head and facial dynamics stay perfectly smooth and continuous."
            ),
        },
    },
    {
        "video_path": "TVSum/Round3/videos/stitched/test/PJrm840pAUI-0_00-21_43.mp4",
        "name": "相机镜头(PJrm840pAUI)",
        "captions": {
            "good": (
                "The lens barrel shows inconsistent curvature and edge warping with unnatural transitions. "
                "The camera's grip shifts abruptly from a stable hold to a rotated, unnatural tilt. "
                "Before the transition the camera body is soft with muted texture, then its lens ring and buttons suddenly sharpen with realistic detail."
            ),
            "generic": "There is a fake segment.",
            "contradict": (
                "The lens barrel stays perfectly straight with consistent edges. "
                "The camera's grip remains stable and smoothly angled throughout. "
                "The camera body keeps uniform texture and detail across the whole video."
            ),
        },
    },
    {
        "video_path": "I24V/Round4/videos/stitched/test/P02C05_115902976-part013.mp4",
        "name": "白色轿车(Round4)",
        "captions": {
            "good": "A white car appears suddenly in the upper right of the frame and then vanishes without occlusion or an exit path.",
            "generic": "Something is wrong.",
            "contradict": "The white car moves smoothly along a continuous trajectory and remains visible throughout.",
        },
    },
]


def load_test_annotations(anno_path: pathlib.Path) -> list:
    if not anno_path.exists():
        raise FileNotFoundError(f"找不到测试标注：{anno_path}\n请用 --annotation 指定。")
    with anno_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, list), "标注顶层应是 list"
    return data


# ---------------------------------------------------------------------------
# 3. 详细打分：复用正式 judge 的每一步，但把中间量（facts/claims/矩阵/匹配）也暴露出来
# ---------------------------------------------------------------------------
def detailed_score(judge, caption: str, evidence) -> dict:
    """复刻 EntailmentExplanationJudge.score 的内部步骤，返回全部中间量。

    这样既能对照正式 reward，又能把 NLI 配对矩阵、一对一匹配结果摊开看。
    """
    facts = text_reward.evidence_facts(evidence)          # 证据事实（带权重）
    claims = text_reward._split_claims(caption)           # 原子 claim

    # (fact, claim) 组合 —— 顺序是 claim 外循环、fact 内循环，与正式实现一致
    pairs = [(fact.matching_text, claim) for claim in claims for fact in facts]
    probs = judge.nli.probabilities(pairs)                 # [(entailment, contradiction), ...]

    entailment = [[0.0 for _ in facts] for _ in claims]
    contradiction = [[0.0 for _ in facts] for _ in claims]
    neutral = [[0.0 for _ in facts] for _ in claims]
    cursor = 0
    for ci in range(len(claims)):
        for fi in range(len(facts)):
            e, c = probs[cursor]
            entailment[ci][fi] = e
            contradiction[ci][fi] = c
            neutral[ci][fi] = max(0.0, 1.0 - e - c)        # softmax 三分类，neutral 是余量
            cursor += 1

    aligned = text_reward._maximum_weight_matching(entailment)  # [(claim_idx, fact_idx, entailment)]

    # 精确复用正式公式
    precision = sum(s for _, _, s in aligned) / len(claims) if claims else 0.0
    total_weight = sum(f.weight for f in facts)
    aligned_by_fact = {fi: s for _, fi, s in aligned}
    recall = (
        sum(f.weight * aligned_by_fact.get(fi, 0.0) for fi, f in enumerate(facts)) / total_weight
        if total_weight else 0.0
    )
    graph_f1 = text_reward.EntailmentExplanationJudge._weighted_f1(precision, recall)
    contradiction_score = (
        sum(contradiction[ci][fi] for ci, fi, _ in aligned) / len(claims) if claims else 0.0
    )
    reward = 0.55 * recall + 0.45 * precision - 0.50 * contradiction_score
    reward = max(-1.0, min(1.0, reward))

    return {
        "facts": [{"relation": f.relation, "weight": f.weight, "matching_text": f.matching_text} for f in facts],
        "claims": claims,
        "entailment": entailment,
        "contradiction": contradiction,
        "neutral": neutral,
        "aligned": [{"claim_idx": ci, "fact_idx": fi, "entailment": s} for ci, fi, s in aligned],
        "precision": precision,
        "recall": recall,
        "graph_f1": graph_f1,
        "contradiction_score": contradiction_score,
        "reward": reward,
        # 对照：正式 judge 给出的 reward，应当与上面 reward 一致（会再跑一次 NLI）
        "official_reward": judge.score(caption=caption, evidence=evidence).reward,
    }


# ---------------------------------------------------------------------------
# 4. 输出：markdown 报告 + json 原始数据
# ---------------------------------------------------------------------------
def _fmt(p: float) -> str:
    return f"{p:.3f}"


def build_report(samples, device: str, nli_model: pathlib.Path, annotation: pathlib.Path) -> str:
    lines: list[str] = []
    lines.append("# NLI 文字解释奖励 —— 演示报告")
    lines.append("")
    lines.append(f"- 测试标注：`{annotation}`")
    lines.append(f"- NLI 模型：`{nli_model}`（microsoft/deberta-v3-small 微调的 NLI，冻结）")
    lines.append(f"- 设备：`{device}`")
    lines.append(f"- 打分实现：仓库 `Trace/trace/text_explanation_reward.py` 的 `EntailmentExplanationJudge`")
    lines.append("")
    lines.append("## 公式")
    lines.append("")
    lines.append("```text")
    lines.append("coverage(recall) = 带权事实召回率（对象异常权重 2，开始/结束各 1）")
    lines.append("precision        = 生成 claim 中有事实支持的比例")
    lines.append("contradiction    = 已匹配 claim/fact 对的矛盾概率均值")
    lines.append("reward = 0.55*recall + 0.45*precision - 0.50*contradiction")
    lines.append("```")
    lines.append("")

    for smp in samples:
        name = smp["name"]
        ann = smp["_ann"]
        details = smp["_details"]
        lines.append(f"## 样例：{name}")
        lines.append("")
        lines.append(f"- 视频：`{ann['video_path']}`")
        lines.append(f"- 伪造区间(segment)：`{ann['annotations'][0]['segment']}`")
        lines.append(f"- 伪造模型：`{ann['annotations'][0]['model']}`")
        lines.append("")
        lines.append("### 证据事实（evidence facts）")
        lines.append("")
        lines.append("| relation | weight | matching_text（label + text） |")
        lines.append("|---|---|---|")
        for f in details["facts"]:
            lines.append(f"| `{f['relation']}` | {f['weight']:g} | {f['matching_text']} |")
        lines.append("")

        for variant in ("good", "generic", "contradict"):
            d = details[variant]
            lines.append(f"### 模拟解释（{variant}）")
            lines.append("")
            lines.append(f"> 模型生成 caption：`{smp['captions'][variant]}`")
            lines.append("")
            lines.append("原子 claim 切分结果：")
            lines.append("")
            for i, c in enumerate(d["claims"]):
                lines.append(f"- claim[{i}]：`{c}`")
            lines.append("")
            lines.append("#### NLI 配对矩阵（行 = claim，列 = fact；E=entailment / C=contradiction / N=neutral）")
            lines.append("")
            header = "| claim \\ fact |" + " | ".join(f"fact[{i}]({f['relation']})" for i, f in enumerate(d["facts"])) + " |"
            sep = "|---|" + "|".join(["---"] * len(d["facts"])) + "|"
            lines.append(header)
            lines.append(sep)
            for ci in range(len(d["claims"])):
                cells = []
                for fi in range(len(d["facts"])):
                    e, c, n = d["entailment"][ci][fi], d["contradiction"][ci][fi], d["neutral"][ci][fi]
                    cells.append(f"E={_fmt(e)} C={_fmt(c)} N={_fmt(n)}")
                lines.append(f"| claim[{ci}] | " + " | ".join(cells) + " |")
            lines.append("")
            lines.append("一对一最大权匹配（按 entailment 选，保证不重复吃分）：")
            lines.append("")
            if d["aligned"]:
                for a in d["aligned"]:
                    frel = d["facts"][a["fact_idx"]]["relation"]
                    lines.append(f"- claim[{a['claim_idx']}] ↔ fact[{a['fact_idx']}]({frel})，entailment={_fmt(a['entailment'])}")
            else:
                lines.append("- （无匹配）")
            lines.append("")
            lines.append("#### 得分")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|---|---|")
            lines.append(f"| precision | {_fmt(d['precision'])} |")
            lines.append(f"| recall(coverage) | {_fmt(d['recall'])} |")
            lines.append(f"| graph_f1 | {_fmt(d['graph_f1'])} |")
            lines.append(f"| contradiction | {_fmt(d['contradiction_score'])} |")
            lines.append(f"| **reward** | **{_fmt(d['reward'])}**（正式 judge：{_fmt(d['official_reward'])}） |")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="NLI 文字解释奖励演示")
    parser.add_argument("--trace-dir", type=pathlib.Path, default=DEFAULT_TRACE_DIR, help="msloc_code/Trace 目录")
    parser.add_argument("--nli-model", type=pathlib.Path, default=DEFAULT_NLI_MODEL, help="冻结 NLI 模型目录")
    parser.add_argument("--annotation", type=pathlib.Path, default=DEFAULT_ANNOTATION, help="测试标注 json")
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="auto=有 GPU 用 GPU")
    args = parser.parse_args()

    # Windows 控制台中文不乱码（Linux 无影响）
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 复用仓库正式实现（EvidenceCard + 证据切分/打分）
    global opd, text_reward
    opd = _load_module("opd_grpo_case", args.trace_dir / "trace" / "opd_grpo.py")
    text_reward = _load_module("text_explanation_reward_case", args.trace_dir / "trace" / "text_explanation_reward.py")

    # 设备选择
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    annos = load_test_annotations(args.annotation)
    by_path = {item["video_path"]: item for item in annos}
    for smp in SAMPLES:
        if smp["video_path"] not in by_path:
            raise KeyError(f"测试标注里没有 {smp['video_path']}")
        smp["_ann"] = by_path[smp["video_path"]]

    # 只加载一次冻结 NLI 模型
    nli = text_reward.FrozenNLIScorer(str(args.nli_model), device=device, batch_size=32)
    judge = text_reward.EntailmentExplanationJudge.__new__(text_reward.EntailmentExplanationJudge)
    judge.require_candidate_observable = False
    judge.nli = nli

    print(f"NLI 模型：{args.nli_model}")
    print(f"设备：{device}")
    print(f"样例数：{len(SAMPLES)}")
    print("=" * 100)

    for smp in SAMPLES:
        ann = smp["_ann"]
        annotation = ann["annotations"][0]
        # 用正式实现从标注构建证据卡（含 Round4 只保留 object 的逻辑）
        evidence = opd.EvidenceCard.from_annotation(annotation)
        facts = text_reward.evidence_facts(evidence)

        print(f"\n【样例】{smp['name']}")
        print(f"  视频：{smp['video_path']}")
        print(f"  segment={annotation['segment']}  model={annotation['model']}")
        print("  证据事实：")
        for f in facts:
            print(f"    - [{f.relation}] weight={f.weight:g}")
            print(f"        {f.matching_text}")

        details = {
            "facts": [{"relation": f.relation, "weight": f.weight, "matching_text": f.matching_text} for f in facts],
        }
        for variant, caption in smp["captions"].items():
            d = detailed_score(judge, caption, evidence)
            details[variant] = d

            print(f"\n  --- 模拟解释({variant}) ---")
            print(f"  caption: {caption}")
            print(f"  claims: {d['claims']}")
            for ci in range(len(d["claims"])):
                for fi in range(len(facts)):
                    e, c, n = d["entailment"][ci][fi], d["contradiction"][ci][fi], d["neutral"][ci][fi]
                    print(f"    claim[{ci}] x fact[{fi}]({facts[fi].relation})  E={e:.3f} C={c:.3f} N={n:.3f}")
            matched = [(a["claim_idx"], details["facts"][a["fact_idx"]]["relation"], round(a["entailment"], 3)) for a in d["aligned"]]
            print(f"  匹配: {matched}")
            print(f"  precision={d['precision']:.3f} recall={d['recall']:.3f} f1={d['graph_f1']:.3f} "
                  f"contradiction={d['contradiction_score']:.3f}")
            print(f"  => reward={d['reward']:.3f} (official={d['official_reward']:.3f})")

        smp["_details"] = details

    # 写输出到 MSLoc_data
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "nli_case_report.md"
    report_path.write_text(build_report(SAMPLES, device, args.nli_model, args.annotation), encoding="utf-8")

    # json 原始数据（去掉内部引用 _ann/_details 的环，只保留需要的数据）
    json_results = []
    for smp in SAMPLES:
        ann = smp["_ann"]
        annotation = ann["annotations"][0]
        json_results.append({
            "name": smp["name"],
            "video_path": smp["video_path"],
            "segment": annotation["segment"],
            "combine_dir": annotation["combine_dir"],
            "evidence": opd.EvidenceCard.from_annotation(annotation).as_dict(),
            "variants": {v: smp["_details"][v] for v in ("good", "generic", "contradict")},
        })
    json_path = args.out_dir / "nli_case_results.json"
    json_path.write_text(json.dumps(json_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"报告已写入：{report_path}")
    print(f"原始数据：{json_path}")


if __name__ == "__main__":
    main()
