#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NLI 文字解释奖励 —— 真实推理版演示脚本

用 TRACE **基础权重**（trace-uni，暂无 SFT 学生模型）对 2~3 个测试样例做真实推理，
解析出模型**真实生成**的 caption，再喂冻结 NLI 详细打分。

同时打印 caption 的原子切分结果（_split_claims），直接回答一个问题：
"模型输出里到底有没有句号 / 分号"——它决定了解释奖励里"原子 claim"这一步
在真实输出上到底怎么生效。这里**不手写任何模型回答**，全部来自真实生成。

打分逻辑（atomic-entailment-v3，已删 repetition/over_length）：
    reward = 0.55*recall + 0.45*precision - 0.50*contradiction
    - precision：匹配上的 claim 的 entailment 求和 ÷ claim 总数
    - recall   ：Σ(事实权重 × 匹配 entailment) ÷ Σ 权重（object=2, onset=1, offset=1）
    - contradiction：已匹配对的 contradiction 求和 ÷ claim 总数

推理链路照搬仓库 `Trace/trace/eval/evaluate_ref.py`：
    加载模型（覆盖 config 里遗留的 vision_tower 路径）→ process_video_ref_split 抽帧 → generate → parse_trace_tokens 解析 caption。

运行（远程服务器，在 msloc_code 根目录；GPU + trace-uni 权重 + 视频 + NLI 模型）：
    python case/test_nli_reward.py \
        --model-path ../MSLoc_data/Trace/ckpts/trace-uni \
        --vision-tower ../MSLoc_data/Trace/ckpts/clip-vit-large-patch14-336 \
        --video-root ../MSLoc_data/data/Tasle-CoT-10K/videos \
        --annotation ../MSLoc_data/data/Tasle-CoT-10K/annos/test_all_1209_0119.json \
        --nli-model ../MSLoc_data/Trace/ckpts/nli-deberta-v3-small \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# ---------------------------------------------------------------------------
# 路径定位（相对脚本位置；远程目录结构同本地仓库约定）
# ---------------------------------------------------------------------------
CASE_DIR = pathlib.Path(__file__).resolve().parent            # msloc_code/case
CODE_ROOT = CASE_DIR.parent                                   # msloc_code
TRACE_ROOT = CODE_ROOT / "Trace"                              # 复用仓库正式实现
DATA_ROOT = CODE_ROOT.parent / "MSLoc_data"

DEFAULT_VIDEO_ROOT = DATA_ROOT / "data" / "Tasle-CoT-10K" / "videos"
DEFAULT_ANNOTATION = DATA_ROOT / "data" / "Tasle-CoT-10K" / "annos" / "test_all_1209_0119.json"
DEFAULT_NLI_MODEL = DATA_ROOT / "Trace" / "ckpts" / "nli-deberta-v3-small"
DEFAULT_PROMPT_FILE = TRACE_ROOT / "trace" / "prompts" / "dvc.txt"
DEFAULT_OUT_DIR = DATA_ROOT / "Trace" / "output" / "nli_case_demo"

# 让 `trace` 包可 import（Trace/trace/__init__.py）
if str(TRACE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACE_ROOT))

# 这些模块顶层只 import 标准库，安全；torch/transformers 在函数内按需加载
from trace.opd_grpo import (  # noqa: E402
    TraceTokenSpec, parse_trace_tokens, EvidenceCard,
    VALID_EVENT, VALID_NO_EVENT, FORMAT_FAILURE,
)
from trace.text_explanation_reward import (  # noqa: E402
    evidence_facts, _split_claims, _maximum_weight_matching,
    FrozenNLIScorer, EntailmentExplanationJudge,
)

# ---------------------------------------------------------------------------
# 样例：选 3 个标注齐全的假视频（2 个 Round3 = 对象+开始+结束三证据；1 个 Round4 = 只有对象）
# 它们的 GT segment 会作为 proposal 喂给模型（即让模型看真实伪造片段）。
# ---------------------------------------------------------------------------
SAMPLE_KEYS = [
    "TVSum/Round3/videos/stitched/test/gzDbaEs1Rlg-20_57-96_06.mp4",
    "TVSum/Round3/videos/stitched/test/PJrm840pAUI-0_00-21_43.mp4",
    "I24V/Round4/videos/stitched/test/P02C05_115902976-part013.mp4",
]


def safe_decode_text(tokenizer, token_ids):
    """只 decode 基础文本 token；生成的 sync/time/score id 不是 SentencePiece id。"""
    base_vocab_size = getattr(tokenizer, "vocab_size", None)
    filtered_ids = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id < 0:
            continue
        if base_vocab_size is not None and token_id >= base_vocab_size:
            continue
        filtered_ids.append(token_id)
    return tokenizer.decode(filtered_ids, skip_special_tokens=True) if filtered_ids else ""


def read_txt(path):
    with open(path, "r", encoding="utf-8") as fin:
        return fin.readline().strip()


def load_trace_model(model_path: str, vision_tower_path: str, device):
    """加载 TRACE 模型，并覆盖 config 里训练机遗留的 vision tower 路径。

    仓库的 load_pretrained_model 会把 --vision-tower 塞进 **kwargs 后静默忽略
    （TraceMistralForCausalLM.__init__(config, **kwargs) 不下传），实际读的是
    config.json 里的 mm_vision_tower。这里照搬 train_mt.py 的 apply_runtime_model_config：
    先 AutoConfig 读 config → 改 mm_vision_tower → from_pretrained(config=config)。
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from trace.model.language_model.trace_mistral import TraceMistralForCausalLM
    from trace.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.mm_vision_tower = vision_tower_path
    config.vision_tower = vision_tower_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = TraceMistralForCausalLM.from_pretrained(
        model_path, config=config, low_cpu_mem_usage=True,
        torch_dtype=torch.float16, device_map=None,
    )
    model = model.to(device)
    model.to(dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    processor = vision_tower.image_processor

    return tokenizer, model, processor


# ---------------------------------------------------------------------------
# NLI 详细打分：复用正式 judge 的每一步，但把配对矩阵、一对一匹配也摊开返回
# ---------------------------------------------------------------------------
def detailed_nli_score(judge, caption: str, evidence) -> dict:
    facts = evidence_facts(evidence)
    claims = _split_claims(caption)
    facts_info = [{"relation": f.relation, "weight": f.weight, "matching_text": f.matching_text} for f in facts]

    if not facts or not claims:
        return {
            "facts": facts_info, "claims": claims,
            "matrix": None, "aligned": [],
            "precision": 0.0, "recall": 0.0, "graph_f1": 0.0, "contradiction": 0.0,
            "reward": -1.0,
            "official_reward": judge.score(caption=caption, evidence=evidence).reward,
        }

    # (fact, claim) 组合：claim 外循环 × fact 内循环，与正式实现一致
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

    aligned = _maximum_weight_matching(entailment)         # [(claim_idx, fact_idx, entailment)]

    # 精确复用正式公式
    precision = sum(s for _, _, s in aligned) / len(claims)
    total_weight = sum(f.weight for f in facts)
    aligned_by_fact = {fi: s for _, fi, s in aligned}
    recall = (
        sum(f.weight * aligned_by_fact.get(fi, 0.0) for fi, f in enumerate(facts)) / total_weight
        if total_weight else 0.0
    )
    graph_f1 = EntailmentExplanationJudge._weighted_f1(precision, recall)
    contradiction_score = (
        sum(contradiction[ci][fi] for ci, fi, _ in aligned) / len(claims) if claims else 0.0
    )
    reward = 0.55 * recall + 0.45 * precision - 0.50 * contradiction_score
    reward = max(-1.0, min(1.0, reward))

    return {
        "facts": facts_info, "claims": claims,
        "matrix": {"entailment": entailment, "contradiction": contradiction, "neutral": neutral},
        "aligned": [{"claim_idx": ci, "fact_idx": fi, "entailment": s} for ci, fi, s in aligned],
        "precision": precision, "recall": recall, "graph_f1": graph_f1,
        "contradiction": contradiction_score,
        "reward": reward,
        "official_reward": judge.score(caption=caption, evidence=evidence).reward,
    }


# ---------------------------------------------------------------------------
# 输出：markdown 报告
# ---------------------------------------------------------------------------
def _fmt(p) -> str:
    return f"{float(p):.3f}"


def build_report(records, device, nli_device, args) -> str:
    L: list[str] = []
    L.append("# NLI 文字解释奖励 —— 真实推理演示")
    L.append("")
    L.append(f"- 模型：`{args.model_path}`（TRACE 基础权重，未 SFT）")
    L.append(f"- 推理设备：`{device}`　NLI 设备：`{nli_device}`")
    L.append(f"- NLI 模型：`{args.nli_model}`")
    L.append(f"- 测试标注：`{args.annotation}`")
    L.append("")
    L.append("## 公式")
    L.append("")
    L.append("```text")
    L.append("reward = 0.55*recall + 0.45*precision - 0.50*contradiction")
    L.append("```")
    L.append("")

    for rec in records:
        L.append(f"## 样例：{rec['name']}")
        L.append("")
        L.append(f"- 视频：`{rec['video_path']}`")
        L.append(f"- GT 伪造区间：`{rec['segment']}`（作为 proposal 喂给模型）")
        L.append(f"- 伪造模型：`{rec['model']}`")
        L.append("")
        L.append("### 证据事实（GT 标注，NLI 打分的参照）")
        L.append("")
        L.append("| relation | weight | matching_text |")
        L.append("|---|---|---|")
        for f in rec["evidence_facts"]:
            L.append(f"| `{f['relation']}` | {f['weight']:g} | {f['matching_text']} |")
        L.append("")

        g = rec["generation"]
        L.append("### 模型真实输出")
        L.append("")
        L.append(f"- 解析状态：`{g['parse_status']}`")
        if g["failure_reasons"]:
            L.append(f"- 失败原因：`{g['failure_reasons']}`")
        L.append(f"- 生成 token 数：`{g['num_tokens']}`")
        L.append(f"- 解析出的片段：`{g['segments']}`")
        L.append("")
        L.append(f"> 真实 caption：`{g['caption']}`")
        L.append("")

        d = rec["nli"]
        L.append("### caption 原子切分（_split_claims）")
        L.append("")
        if d["claims"]:
            for i, c in enumerate(d["claims"]):
                L.append(f"- claim[{i}]：`{c}`")
        else:
            L.append("- （空 caption，无法切分）")
        L.append("")

        if d["matrix"] is None:
            L.append("### NLI 打分")
            L.append("")
            L.append("空 caption 或空事实 → reward = -1.0")
        else:
            L.append("### NLI 配对矩阵（行=claim，列=fact；E/C/N = entailment/contradiction/neutral）")
            L.append("")
            header = "| claim \\ fact |" + " | ".join(f"fact[{i}]({f['relation']})" for i, f in enumerate(d["facts"])) + " |"
            L.append(header)
            L.append("|---|" + "|".join(["---"] * len(d["facts"])) + "|")
            for ci in range(len(d["claims"])):
                cells = []
                for fi in range(len(d["facts"])):
                    e = d["matrix"]["entailment"][ci][fi]
                    c = d["matrix"]["contradiction"][ci][fi]
                    n = d["matrix"]["neutral"][ci][fi]
                    cells.append(f"E={_fmt(e)} C={_fmt(c)} N={_fmt(n)}")
                L.append(f"| claim[{ci}] | " + " | ".join(cells) + " |")
            L.append("")
            L.append("一对一最大权匹配：")
            L.append("")
            if d["aligned"]:
                for a in d["aligned"]:
                    frel = d["facts"][a["fact_idx"]]["relation"]
                    L.append(f"- claim[{a['claim_idx']}] ↔ fact[{a['fact_idx']}]({frel})，entailment={_fmt(a['entailment'])}")
            else:
                L.append("- （无匹配）")
            L.append("")
            L.append("### NLI 打分")
            L.append("")
            L.append("| 指标 | 值 |")
            L.append("|---|---|")
            L.append(f"| precision | {_fmt(d['precision'])} |")
            L.append(f"| recall(coverage) | {_fmt(d['recall'])} |")
            L.append(f"| graph_f1 | {_fmt(d['graph_f1'])} |")
            L.append(f"| contradiction | {_fmt(d['contradiction'])} |")
            L.append(f"| **reward** | **{_fmt(d['reward'])}**（官方 judge：{_fmt(d['official_reward'])}） |")
        L.append("")
        L.append("---")
        L.append("")
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description="NLI 文字解释奖励 —— 真实推理演示")
    parser.add_argument("--model-path", type=pathlib.Path, required=True, help="TRACE 基础权重目录（trace-uni）")
    parser.add_argument("--vision-tower", type=pathlib.Path, required=True, help="CLIP vision tower 目录")
    parser.add_argument("--video-root", type=pathlib.Path, default=DEFAULT_VIDEO_ROOT, help="视频根目录")
    parser.add_argument("--annotation", type=pathlib.Path, default=DEFAULT_ANNOTATION, help="测试标注 json")
    parser.add_argument("--nli-model", type=pathlib.Path, default=DEFAULT_NLI_MODEL, help="冻结 NLI 模型目录")
    parser.add_argument("--prompt-file", type=pathlib.Path, default=DEFAULT_PROMPT_FILE, help="推理提示词文件")
    parser.add_argument("--device", default="cuda", help="模型推理设备")
    parser.add_argument("--nli-device", default=None, help="NLI 设备，默认同 --device")
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bnd-frames", type=int, default=16)
    parser.add_argument("--seg-frames", type=int, default=8)
    parser.add_argument("--bnd-ratio", type=float, default=0.2)
    parser.add_argument("--num-samples", type=int, default=3, help="推理前 N 个样例")
    args = parser.parse_args()

    # 推理/解析依赖 torch 与 trace 模型，放到运行时再 import（远程有完整环境）
    import torch
    from trace.conversation import conv_templates
    from trace.constants import DEFAULT_MMODAL_TOKEN
    from trace.mm_utils import tokenizer_MMODAL_token_all, process_video_ref_split

    device = torch.device(args.device)
    nli_device = args.nli_device or args.device

    # ---- 加载标注，选样例 ----
    with args.annotation.open("r", encoding="utf-8") as fh:
        annos = json.load(fh)
    by_path = {item["video_path"]: item for item in annos}
    picked = []
    for key in SAMPLE_KEYS[: args.num_samples]:
        if key not in by_path:
            raise KeyError(f"标注里没有 {key}")
        picked.append((key, by_path[key]))

    # ---- 加载模型（覆盖 config 里训练机遗留的 vision tower 路径）----
    model_path = str(args.model_path.resolve())
    vision_tower_path = str(args.vision_tower.resolve())
    tokenizer, model, processor = load_trace_model(model_path, vision_tower_path, device)
    trace_token_spec = TraceTokenSpec(
        text_vocab_size=model.vocab_size,
        time_vocab=model.get_model().time_tokenizer.vocab,
        score_vocab_size=model.config.score_vocab_size,
    )
    prompt = read_txt(str(args.prompt_file))
    print(f"模型加载完成，device={device}，time_vocab={len(trace_token_spec.time_vocab)}")

    # ---- 初始化 NLI judge（只加载一次） ----
    nli = FrozenNLIScorer(str(args.nli_model), device=nli_device, batch_size=32)
    judge = EntailmentExplanationJudge.__new__(EntailmentExplanationJudge)
    judge.require_candidate_observable = False
    judge.nli = nli

    records = []
    for key, ann in picked:
        annotation = ann["annotations"][0]
        segment = annotation["segment"]                     # GT 伪造区间，作为 proposal
        win_s, win_e = float(segment[0]), float(segment[1])
        vid_path = args.video_root / ann["video_path"]
        if not vid_path.exists():
            print(f"[跳过] 视频不存在：{vid_path}")
            continue

        print(f"\n=== 推理 {key}  proposal={segment} ===")
        try:
            tensor, video_timestamps = process_video_ref_split(
                str(vid_path), processor, model.config.image_aspect_ratio,
                bnd_frames=args.bnd_frames, seg_frames=args.seg_frames, bnd_ratio=args.bnd_ratio,
                start_time=win_s, end_time=win_e,
            )
            tensor = tensor.to(dtype=torch.float16, device=device, non_blocking=True)

            default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
            question = default_mm_token + "\n" + prompt
            conv = conv_templates["llama_2"].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            cur_prompt = conv.get_prompt() + "<sync>"
            input_ids = tokenizer_MMODAL_token_all(cur_prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(device)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    images_or_videos=[tensor],
                    modal_list=["video"],
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    video_timestamps=[video_timestamps],
                    heads=[1],
                )

            raw_ids = [int(x) for x in output_ids[0].detach().cpu().tolist()]
            parsed = parse_trace_tokens(
                raw_ids, trace_token_spec,
                lambda ids: safe_decode_text(tokenizer, ids),
                window_duration=win_e - win_s,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[推理失败] {exc}")
            parsed = None

        # ---- 构建 GT 证据卡 + NLI 打分 ----
        evidence = EvidenceCard.from_annotation(annotation)
        caption = parsed.caption if parsed is not None else ""
        nli_detail = detailed_nli_score(judge, caption, evidence)

        generation = {
            "parse_status": parsed.status if parsed else FORMAT_FAILURE,
            "failure_reasons": parsed.failure_reasons if parsed else ["generation_exception"],
            "num_tokens": len(parsed.raw_token_ids) if parsed else 0,
            "segments": parsed.segments if parsed else [],
            "caption": caption,
        }

        print(f"  parse_status={generation['parse_status']}  segments={generation['segments']}")
        print(f"  failure_reasons={generation['failure_reasons']}")
        print(f"  真实 caption: {caption!r}")
        print(f"  claims({len(nli_detail['claims'])}): {nli_detail['claims']}")
        print(f"  NLI: precision={nli_detail['precision']:.3f} recall={nli_detail['recall']:.3f} "
              f"contradiction={nli_detail['contradiction']:.3f} => reward={nli_detail['reward']:.3f} "
              f"(official={nli_detail['official_reward']:.3f})")

        records.append({
            "name": key,
            "video_path": ann["video_path"],
            "segment": segment,
            "model": annotation["model"],
            "combine_dir": annotation["combine_dir"],
            "evidence_facts": nli_detail["facts"],
            "evidence": evidence.as_dict(),
            "generation": generation,
            "nli": nli_detail,
        })

    if not records:
        raise SystemExit("没有成功推理任何样例，请检查视频路径与模型。")

    # ---- 写输出 ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "nli_case_report.md"
    report_path.write_text(build_report(records, str(device), nli_device, args), encoding="utf-8")
    json_path = args.out_dir / "nli_case_results.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告：{report_path}")
    print(f"原始数据：{json_path}")


if __name__ == "__main__":
    main()
