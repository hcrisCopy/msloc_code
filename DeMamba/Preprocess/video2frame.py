import os
import argparse
from glob import glob
import fnmatch
from moviepy.editor import VideoFileClip
import multiprocessing
import math
import subprocess

# Defaults are RELATIVE to the MSLoc repository root.
# You can also pass --input_root / --output_root from the CLI.
DEFAULT_INPUT_ROOT = '../../MSLoc_data/data/Tasle-CoT-10K/videos'
DEFAULT_OUTPUT_ROOT = '../../MSLoc_data/data/Tasle-CoT-10K/video_frames'

INPUT_ROOT = DEFAULT_INPUT_ROOT
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT

def find_video_files(root_dir, extensions=['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']):
    """Recursively collect every video file under root_dir."""
    video_files = []
    for extension in extensions:
        pattern = f"**/{extension}" if os.name != 'nt' else f"**\\{extension}"
        video_files.extend(glob(os.path.join(root_dir, pattern), recursive=True))
    return video_files

def get_video_length(file_path):
    try:
        video = VideoFileClip(file_path)
        duration = video.duration
        video.close()
        return duration
    except Exception as e:
        print(f"Failed to read duration for {file_path}: {str(e)}")
        return 0

def process_video(video_path):
    # Video name without extension.
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Path of `video_path` relative to INPUT_ROOT.
    relative_path = os.path.relpath(os.path.dirname(video_path), INPUT_ROOT)

    # Mirror the directory structure under OUTPUT_ROOT.
    image_path = os.path.join(OUTPUT_ROOT, relative_path, video_name)

    # Skip if already processed.
    if os.path.exists(image_path) and len(os.listdir(image_path)) > 0:
        return

    os.makedirs(image_path, exist_ok=True)

    video_length = get_video_length(video_path)
    if video_length == 0:
        print(f"Skipping invalid video: {video_path}")
        return

    inter_val = 8
    # ``image_path`` is absolute (see main below).  Do not change the process
    # CWD: with a relative output path, the old implementation accidentally
    # constructed a second nested output path after ``chdir`` and ffmpeg
    # failed with exit code 1 (reported by os.system as 256).
    output_pattern = os.path.join(image_path, "frame_%d.jpg")
    result = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path,
         "-r", str(inter_val), output_pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(f"ffmpeg failed ({result.returncode}): {video_path}\n{result.stderr.strip()}")
    else:
        print(f"Frames extracted: {video_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, default=DEFAULT_INPUT_ROOT,
                        help='Root directory of input videos (relative or absolute).')
    parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT,
                        help='Root directory to save extracted frames.')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    # Resolve paths before worker processes start.  This also makes every
    # output pattern valid regardless of a worker's current directory.
    INPUT_ROOT = os.path.abspath(args.input_root)
    OUTPUT_ROOT = os.path.abspath(args.output_root)

    print(f"Extracting video frames... input={INPUT_ROOT} output={OUTPUT_ROOT}")

    video_files = find_video_files(INPUT_ROOT)
    print(f"Found {len(video_files)} video file(s)")

    pool = multiprocessing.Pool(processes=args.num_workers)
    pool.map(process_video, video_files)
    pool.close()
    pool.join()
    print("All videos processed.")
