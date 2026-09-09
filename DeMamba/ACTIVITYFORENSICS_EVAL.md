# ActivityForensics raw-video evaluation

This evaluator tests the existing, trained first-stage `XCLIP_NeuronDeMamba_4`
checkpoint without fine-tuning it.  It is therefore a **zero-shot external
generalization** experiment, not the ActivityForensics paper's trained TADiff
setting.

## 1. Download

The benchmark's Hugging Face release contains raw video files and CSV metadata.
This project consumes video pixels, so download the full raw release, not only
the feature archive.  In PowerShell, from the repository root:

```powershell
python -m pip install -U huggingface_hub
hf auth login                    # public files normally do not need this; run it if HF asks
hf download ActivityForensics/ActivityForensics --repo-type dataset --local-dir ..\MSLoc_data\ActivityForensics
```

The full raw release is large (about 120 GB), so ensure enough free disk space.
The authors also provide a OneDrive download (password `ActivityForensics`) in
their official repository.  Do not use only `feat/`: those CLIP-L14 feature
files cannot be input to this project's frozen XCLIP patch encoder.

Expected paths after download are conceptually:

```text
..\MSLoc_data\ActivityForensics\
  ... test*.csv (HF metadata; has a file_name column)
  video\... raw .mp4 files ...
```

The script searches `--video-root` recursively, so the exact video subfolder
layout in a particular release does not matter.  The authors' separate
`annot/test@*.txt` layout is also supported with
`--annotation-source official-txt`.

## 2. What the annotation means

The authors' native annotation loader parses a fake-video line as:

```text
video.mp4 <duration_seconds> <start>=<end>+<start>=<end>
```

For example, `A.mp4 29.4 2.10=8.20` has one manipulated interval.  `+` denotes
multiple independent intervals.  A real-video line has only
`video.mp4 <duration_seconds>`.

The Hugging Face release represents the same labels differently: every row's
`file_name` is the raw path, and its filename embeds the intervals, e.g.
`video/02_wan/00ZCA+10.50=16.40=...+24.90=42.40=...mp4`.  The evaluator extracts
every `+start=end=` pair and reads the video duration from the MP4 itself.
Thus the CSV is a label file, not merely a list of files.  Temporal endpoints
are in seconds.  The official config evaluates six fake test splits
(`test@wan`, `test@scifi`, `test@fcvg`, `test@vace-1.3B`, `test@ltx`,
`test@vidu`); it reports AP at tIoU .75/.85/.95 and AR@1/@5/@10.

## 3. Run the current DeMamba model

The Hugging Face snapshot stores `metadata/test.csv` and the raw MP4 files
directly, so it does not need extraction. Run from the repository root. The
current checkpoint, config, annotation, video, and output paths are all the
script's relative defaults:

```powershell
python DeMamba/eval_activityforensics.py --clean
```

With a native `annot/test@*.txt` download instead, replace the annotation
arguments with `--annotation-dir ..\MSLoc_data\ActivityForensics\annot
--annotation-source official-txt --annotation-pattern "test@*.txt"`.

By default the evaluator includes Wan, SciFi, FCVG, VACE, LTX, and Vidu. It
reports Wan/VACE/LTX/Vidu as in-domain and SciFi/FCVG as out-of-domain.
It validates and prints the available/missing video counts and the available
window count before loading the model. A partial download is supported: missing
or unreadable videos are written to `skipped_videos.json`, and metrics are
computed only over videos that actually complete inference. Once the download
is complete, the same command automatically evaluates the full set.

Outputs:

- `window_predictions.json`: four-class probabilities for every two-second
  window and the merged proposals.
- `dataset_manifest.json`: the preflight-validated video list and counts.
- `skipped_videos.json`: annotated videos not yet downloaded or currently
  unreadable (only produced for a partial snapshot).
- `predictions.json`: TASLE/evaluate_long-compatible prediction records.
- `metrics.json` / `metrics.csv`: TASLE Det_Acc/F1Det/F1Loc and
  ActivityForensics AP/AR for all, ID, OOD, and every generator.
- `detailed_results.csv`: one row per evaluated video.

`--clean` removes only the evaluator's known output files. Without it, those
files are overwritten while unrelated files in the output directory remain.

The default window is 2 seconds with 8 frames and stride 2 seconds, inherited
from your training YAML.  The foreground rule is exactly the current evaluator:
`max(P(fake), P(r2f), P(f2r)) > P(real)`.  Adjacent foreground windows are
merged; any predicted real window splits them into separate proposals.  It does
not add an undocumented r2f/f2r state machine or use ground truth to alter a
proposal.

This is intentionally conservative, but 2-second non-overlapping windows give
coarse boundaries.  AP@.95 should be treated as a boundary-stress result, not
compared directly with a detector trained on ActivityForensics or using a
continuous boundary regressor.

## 4. Seen/unseen must be stated with the reference set

There are two different meanings of `seen` here.

| ActivityForensics generator | In TASLE training set behind current DeMamba? | In ActivityForensics authors' open-world training? |
|---|---:|---:|
| Wan | Yes (`Wan2.1`) | Seen |
| SciFi | No | Seen |
| FCVG | No | Seen |
| VACE | Yes | Seen |
| LTX | Yes (`LTXVideo`) | Seen |
| Vidu | Yes (`vidu`) | Unseen |

The first column comes from TASLE Table 9: its generators are Wan2.1, Kling,
vidu, Jimeng (FLF2V); LTXVideo, SkyReels-V1, Hailuo, Kling (TI2V); and VACE
(MV2V).  Consequently, in **your zero-shot evaluation**, SciFi and FCVG are
direct generator-family unseen; Wan/VACE/LTX/Vidu overlap TASLE generators.
The `Vidu = unseen` label belongs only to ActivityForensics' own fine-tuning
protocol, where they trained on Wan/SciFi/FCVG/VACE/LTX and held out Vidu.  It
does *not* make Vidu unseen for the current TASLE-trained model.

Report both the full six-split aggregate and at least these groups: direct-tool
overlap (Wan, VACE, LTX, Vidu) versus direct-tool-unseen (SciFi, FCVG).  Also
show each tool separately; Vidu should be identified as a special case with
opposite seen/unseen status under the two protocols.
