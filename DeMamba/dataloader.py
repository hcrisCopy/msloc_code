import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import albumentations
import random
import os
import numpy as np
import cv2
import math
import warnings
import json
import hashlib
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm

def get_video_fake_segments(json_data: List[Dict]) -> Dict[str, Dict]:
    """
    Build a per-video summary containing the fake segments and the video type.

    Args:
        json_data: Raw JSON data with video info and annotations.

    Returns:
        A dict mapping video name to {fake_segments, type, ...}.
    """
    video_info_dict = {}
    
    for video_item in json_data:
        video_path = video_item["video_path"]
        video_type = video_item["type"]
        duration = video_item["duration"]
        annotations = video_item.get("annotations", [])
        
        # Strip the .mp4 extension and replace path separators with underscores.
        video_name = Path(video_path).stem
        formatted_video_name = video_path[:-4].replace('/', '_')
        
        # Collect fake-segment intervals.
        fake_segments = []
        tool = 'None'
        for ann in annotations:
            if "segment" in ann and len(ann["segment"]) == 2 and video_type == "fake":
                start_time, end_time = ann["segment"]
                fake_segments.append((start_time, end_time))
                tool = ann["model"] if (tool in ["None", "Wan2.1"]) else tool
        
        # If no explicit fake segments are listed but the type is fake, use the whole duration.
        if not fake_segments and video_type == "fake":
            fake_segments = [(-1, duration)]

        video_info_dict[formatted_video_name] = {
            "type": video_type,
            "duration": duration,
            "fake_segments": fake_segments,
            "tool": tool,
            "real_fake_segments": [x['segment'] for x in annotations]
        }
    
    return video_info_dict

def parse_json_to_windows(json_data: List[Dict], dataset_base_path: str, 
                          window_length: float = 1.0, frames_per_window: int = 4,
                          mode: str = "binary", cache_dir: str = None,
                          progress_desc: str = "Loading video windows") -> List[Dict]:
    """
    Parse JSON data and produce sliding-window samples.

    Args:
        json_data: Raw JSON data.
        dataset_base_path: Dataset root path.
        window_length: Window length in seconds.
        frames_per_window: Number of frames sampled per window.
        mode: Classification mode, "binary" or "four_class".

    Returns:
        A list of window dicts containing frame paths and labels.
    """
    if cache_dir:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        windows = []
        hits = 0
        iterator = tqdm(json_data, desc=progress_desc, unit="video", dynamic_ncols=True)
        for video_item in iterator:
            cache_key = hashlib.sha1(json.dumps({
                "video": video_item,
                "dataset_base_path": str(Path(dataset_base_path).resolve()),
                "window_length": window_length,
                "frames_per_window": frames_per_window,
                "mode": mode,
            }, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            cache_file = cache_root / f"{cache_key}.json"
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.is_file() else None
            except (OSError, json.JSONDecodeError):
                cached = None
            if isinstance(cached, list):
                video_windows = cached
                hits += 1
            else:
                video_windows = parse_json_to_windows(
                    [video_item], dataset_base_path, window_length, frames_per_window, mode
                )
                temporary = cache_file.with_suffix(".tmp")
                temporary.write_text(json.dumps(video_windows, ensure_ascii=False), encoding="utf-8")
                os.replace(temporary, cache_file)
            windows.extend(video_windows)
            iterator.set_postfix(cache_hits=hits)
        return windows

    fps = 8  # fixed frame rate
    windows = []
    
    for video_item in json_data:
        video_path = video_item["video_path"]
        video_type = video_item["type"]
        duration = video_item["duration"]
        annotations = video_item.get("annotations", [])
        
        # Build the frame directory path.
        video_name = Path(video_path).stem  # strip .mp4 suffix
        frame_dir = Path(dataset_base_path) / video_path[:-4]
        new_video_name = video_path[:-4].replace('/', '_')
        
        if not frame_dir.exists():
            warnings.warn(f"Frame directory not found: {frame_dir}")
            continue
        
        # Collect every frame file.
        frame_files = sorted(frame_dir.glob("frame_*.jpg"), key=lambda x: int(x.stem.split('_')[1]))
        if not frame_files:
            warnings.warn(f"No frame files found in: {frame_dir}")
            continue
        
        # Total frames and per-frame duration.
        total_frames = len(frame_files)
        frame_duration = 1.0 / fps  # seconds per frame
        
        # Number of sliding windows.
        num_windows = int(duration // window_length)
        if duration % window_length > 0:
            num_windows += 1
        
        # Time intervals of fake segments.
        fake_segments = []
        for ann in annotations:
            if "segment" in ann and len(ann["segment"]) == 2 and video_type == "fake":
                start_time, end_time = ann["segment"]
                fake_segments.append((start_time, end_time))
        if fake_segments == []:
            fake_segments = [(-1, duration)]
        
        # Generate windows.
        for window_idx in range(num_windows):
            window_start_time = window_idx * window_length
            window_end_time = min((window_idx + 1) * window_length, duration)
            
            # Frame range of the current window.
            start_frame_idx = int(window_start_time * fps)
            end_frame_idx = min(int(window_end_time * fps), total_frames - 1)
            
            if start_frame_idx >= total_frames:
                continue
                
            # Sample frames uniformly within the window.
            frame_indices = []
            if end_frame_idx - start_frame_idx + 1 >= frames_per_window:
                # Enough frames -> uniform sampling.
                step = max(1, (end_frame_idx - start_frame_idx) // frames_per_window)
                frame_indices = list(range(start_frame_idx, end_frame_idx + 1, step))[:frames_per_window]
            else:
                # Not enough frames -> repeat the last one.
                frame_indices = list(range(start_frame_idx, end_frame_idx + 1))
                while len(frame_indices) < frames_per_window:
                    frame_indices.append(frame_indices[-1])
                frame_indices = frame_indices[:frames_per_window]
            
            # Resolve frame paths.
            frame_paths = []
            for frame_idx in frame_indices:
                if frame_idx < len(frame_files):
                    frame_paths.append(str(frame_files[frame_idx]))
                else:
                    # Out of range -> use the last frame.
                    frame_paths.append(str(frame_files[-1]))
            
            # Window label.
            if mode == "binary":
                label = determine_binary_label(window_start_time, window_end_time, 
                                             fake_segments, video_type)
            else:  # four_class
                label = determine_four_class_label(window_start_time, window_end_time, 
                                                  fake_segments, video_type)
                # A short fake interval can put both of its boundaries in one
                # window (real -> fake -> real).  It is not representable by
                # the four-class target and must not receive an arbitrary
                # directional label.
                if label is None:
                    continue
            
            window_info = {
                "window_id": f"{new_video_name}_window_{window_idx}",
                "video_name": new_video_name,
                "window_idx": window_idx,
                "start_time": window_start_time,
                "end_time": window_end_time,
                "frame_paths": frame_paths,
                "label": label,
                "original_type": video_type
            }
            windows.append(window_info)
    return windows


def determine_binary_label(window_start: float, window_end: float, 
                          fake_segments: List[Tuple[float, float]], 
                          video_type: str) -> int:
    """
    Compute the binary label for a window.
    """
    if video_type == "real":
        return 0  # real
    
    # Mark as fake if the window overlaps with any fake segment.
    for seg_start, seg_end in fake_segments:
        if not (window_end <= seg_start or window_start >= seg_end):
            return 1  # fake
    
    return 0  # real


def determine_four_class_label(window_start: float, window_end: float, 
                              fake_segments: List[Tuple[float, float]], 
                              video_type: str) -> int:
    """
    Compute the four-class label for a window.
    0: real, 1: fake, 2: real-to-fake, 3: fake-to-real
    """
    if video_type == "real":
        return 0  # real
    
    # Compute how much of the window overlaps with fake segments.
    window_fake_ratio = 0.0
    fake_duration_in_window = 0.0
    window_duration = window_end - window_start
    
    for seg_start, seg_end in fake_segments:
        overlap_start = max(window_start, seg_start)
        overlap_end = min(window_end, seg_end)
        if overlap_start < overlap_end:
            fake_duration_in_window += (overlap_end - overlap_start)
    
    window_fake_ratio = fake_duration_in_window / window_duration
    
    # Decide which label to use.
    if window_fake_ratio == 0.0:
        return 0  # real
    elif window_fake_ratio == 1.0:
        return 1  # fake
    elif window_fake_ratio > 0.5:
        # Mostly fake -> figure out the transition type.
        if window_start < fake_segments[0][0]:  # assume a single transition point
            return 3  # fake-to-real
        else:
            return 1  # mostly fake -> treat as plain fake
    else:
        # Mostly real -> figure out the transition type.
        if window_end > fake_segments[0][1]:  # assume a single transition point
            return 2  # real-to-fake
        else:
            return 0  # mostly real -> treat as plain real


def crop_center_by_percentage(image, percentage):
    height, width = image.shape[:2]

    if width > height:
        left_pixels = int(width * percentage)
        right_pixels = int(width * percentage)
        start_x = left_pixels
        end_x = width - right_pixels
        cropped_image = image[:, start_x:end_x]
    else:
        up_pixels = int(height * percentage)
        down_pixels = int(height * percentage)
        start_y = up_pixels
        end_y = height - down_pixels
        cropped_image = image[start_y:end_y, :]

    return cropped_image


def normalization_params(transform_config: Dict) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return preprocessing statistics selected by the experiment config.

    Frozen ``microsoft/xclip-base-patch16`` should receive CLIP-normalised
    pixels.  Keep ImageNet statistics as the legacy default so existing
    DeMamba checkpoints remain reproducible.
    """
    normalization = transform_config.get("normalization", "imagenet").lower()
    if normalization in {"clip", "xclip"}:
        return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    if normalization in {"imagenet", "dinov3_lvd", "dinov3", "dinov2"}:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    raise ValueError(f"Unsupported normalization: {normalization!r}")


def resize_interpolation(transform_config: Dict) -> int:
    """Return the OpenCV interpolation selected by the experiment config."""
    interpolation = str(transform_config.get("interpolation", "linear")).lower()
    interpolation_map = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "bicubic": cv2.INTER_CUBIC,
    }
    if interpolation not in interpolation_map:
        raise ValueError(f"Unsupported resize interpolation: {interpolation!r}")
    return interpolation_map[interpolation]


class VideoWindowDatasetTrain(Dataset):
    def __init__(self, windows: List[Dict], mode: str = "binary", transform_config: Dict = None):
        """
        Train-time video-window dataset (with balanced sampling).

        Args:
            windows: List of window dicts.
            mode: Classification mode, "binary" or "four_class".
            transform_config: Augmentation configuration.
        """
        self.windows = windows
        self.mode = mode
        self.transform_config = transform_config or {}
        self.image_size = int(self.transform_config.get("image_size", 224))
        if self.image_size < 1:
            raise ValueError("transform_config.image_size must be positive")
        
        # Number of classes is determined by the mode.
        self.num_classes = 2 if mode == "binary" else 4
        
        # Balanced sampling.
        self._balance_samples()
    
    def _balance_samples(self):
        """Balance positive and negative samples."""
        if self.mode == "binary":
            # Binary balancing.
            positive_indices = [i for i, w in enumerate(self.windows) if w['label'] == 1]
            negative_indices = [i for i, w in enumerate(self.windows) if w['label'] == 0]
            
            min_samples = min(len(positive_indices), len(negative_indices))
            self.balanced_indices = []
            self.balanced_indices.extend(random.sample(positive_indices, min_samples))
            self.balanced_indices.extend(random.sample(negative_indices, min_samples))
        else:
            # Four-class balancing.
            class_indices = {0: [], 1: [], 2: [], 3: []}
            for i, w in enumerate(self.windows):
                class_indices[w['label']].append(i)
            
            min_samples = min(len(indices) for indices in class_indices.values())
            self.balanced_indices = []
            self.balanced_indices.extend(random.sample(class_indices[0], min(len(class_indices[0]), len(class_indices[1])+len(class_indices[2])+len(class_indices[3]))))
            self.balanced_indices.extend(random.sample(class_indices[1], len(class_indices[1])))
            self.balanced_indices.extend(random.sample(class_indices[2], len(class_indices[2])))
            self.balanced_indices.extend(random.sample(class_indices[3], len(class_indices[3])))
        
        random.shuffle(self.balanced_indices)
    
    def __len__(self):
        return len(self.balanced_indices)
    
    def __getitem__(self, idx):
        real_idx = self.balanced_indices[idx]
        window_info = self.windows[real_idx]
        
        frame_paths = window_info['frame_paths']
        label = window_info['label']
        select_frame_nums = len(frame_paths)
        
        # Build augmentation pipeline.
        aug_list = [albumentations.Resize(
            self.image_size,
            self.image_size,
            interpolation=resize_interpolation(self.transform_config),
        )]
        
        # Train-time augmentations.
        if random.random() < 0.5:
            aug_list.append(albumentations.HorizontalFlip(p=1.0))
        if random.random() < 0.5:
            quality_score = random.randint(50, 100)
            aug_list.append(albumentations.ImageCompression(quality_lower=quality_score, quality_upper=quality_score))
        if random.random() < 0.3:
            aug_list.append(albumentations.GaussNoise(p=1.0))
        if random.random() < 0.3:
            aug_list.append(albumentations.GaussianBlur(blur_limit=(3, 5), p=1.0))
        if random.random() < 0.001:
            aug_list.append(albumentations.ToGray(p=1.0))
        
        norm_mean, norm_std = normalization_params(self.transform_config)
        aug_list.append(albumentations.Normalize(mean=norm_mean,
                                               std=norm_std,
                                               max_pixel_value=255.0, p=1.0))
        trans = albumentations.Compose(aug_list)
        
        # Read and preprocess frames.
        frames = []
        for frame_path in frame_paths:
            try:
                if os.path.exists(frame_path):
                    image = cv2.imread(frame_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Apply optional center-crop.
                    if self.transform_config.get('crop_youku', False) and 'youku' in frame_path:
                        image = crop_center_by_percentage(image, 0.15)
                    
                    augmented = trans(image=image)
                    image = augmented["image"]
                    frames.append(image.transpose(2, 0, 1)[np.newaxis, :])
                else:
                    # Missing frame -> zero padding.
                    frames.append(np.zeros((1, 3, self.image_size, self.image_size), dtype=np.float32))
            except Exception as e:
                warnings.warn(f"Error loading frame {frame_path}: {e}")
                frames.append(np.zeros((1, 3, self.image_size, self.image_size), dtype=np.float32))

        if len(frames) != select_frame_nums:
            raise RuntimeError(f"Expected {select_frame_nums} frames, got {len(frames)}")
        
        # Build the label.
        label_onehot = [0] * self.num_classes
        label_onehot[label] = 1
        
        # Convert to tensors.
        frames = np.concatenate(frames, 0)
        frames = torch.tensor(frames[np.newaxis, :])
        label_onehot = torch.FloatTensor(label_onehot)
        binary_label = torch.FloatTensor([label])
        
        return real_idx, frames, label_onehot, binary_label


class VideoWindowDatasetTest(Dataset):
    def __init__(self, windows: List[Dict], mode: str = "binary", transform_config: Dict = None, task: str = "normal"):
        """
        Test-time video-window dataset (no balanced sampling; evaluate all samples).

        Args:
            windows: List of window dicts.
            mode: Classification mode, "binary" or "four_class".
            transform_config: Augmentation configuration.
            task: Task type, used for task-specific preprocessing.
        """
        self.windows = windows
        self.mode = mode
        self.transform_config = transform_config or {}
        self.image_size = int(self.transform_config.get("image_size", 224))
        if self.image_size < 1:
            raise ValueError("transform_config.image_size must be positive")
        self.task = task
        
        # Number of classes is determined by the mode.
        self.num_classes = 2 if mode == "binary" else 4
        
        # Use every sample at test time (no balancing).
        self.indices = list(range(len(windows)))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        window_info = self.windows[real_idx]
        
        frame_paths = window_info['frame_paths']
        label = window_info['label']
        video_name = window_info['video_name']
        window_idx = window_info['window_idx']
        select_frame_nums = len(frame_paths)
        
        # Build the augmentation pipeline (no random augmentations at test time).
        aug_list = [albumentations.Resize(
            self.image_size,
            self.image_size,
            interpolation=resize_interpolation(self.transform_config),
        )]
        
        norm_mean, norm_std = normalization_params(self.transform_config)
        aug_list.append(albumentations.Normalize(mean=norm_mean,
                                               std=norm_std,
                                               max_pixel_value=255.0, p=1.0))
        trans = albumentations.Compose(aug_list)
        
        # Read and preprocess frames.
        frames = []
        for frame_path in frame_paths:
            try:
                if os.path.exists(frame_path):
                    image = cv2.imread(frame_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Apply optional center-crop.
                    if self.transform_config.get('crop_youku', False) and 'youku' in frame_path:
                        image = crop_center_by_percentage(image, 0.15)
                    
                    augmented = trans(image=image)
                    image = augmented["image"]
                    frames.append(image.transpose(2, 0, 1)[np.newaxis, :])
                else:
                    # Missing frame -> zero padding.
                    frames.append(np.zeros((1, 3, self.image_size, self.image_size), dtype=np.float32))
            except Exception as e:
                warnings.warn(f"Error loading frame {frame_path}: {e}")
                frames.append(np.zeros((1, 3, self.image_size, self.image_size), dtype=np.float32))

        if len(frames) != select_frame_nums:
            raise RuntimeError(f"Expected {select_frame_nums} frames, got {len(frames)}")
        
        # Build the label.
        label_onehot = [0] * self.num_classes
        label_onehot[label] = 1
        
        # Convert to tensors.
        frames = np.concatenate(frames, 0)
        frames = torch.tensor(frames[np.newaxis, :])
        label_onehot = torch.FloatTensor(label_onehot)
        binary_label = torch.FloatTensor([label])
        return window_idx, frames, label_onehot, binary_label, video_name


def generate_dataset_loader_from_json(cfg, evaluation_only=False):
    """
    Build train / test data loaders from JSON annotation files.

    Args:
        cfg: Config dict with keys:
            - train_json_path:    Train annotation JSON.
            - test_json_path:     Test annotation JSON.
            - dataset_base_path:  Dataset root path.
            - window_length:      Window length in seconds.
            - frames_per_window:  Frames sampled per window.
            - mode:               Classification mode.
            - task:               Task type.
            - other training-related parameters.
    """
    # Load JSON files.
    if evaluation_only:
        train_json = []
    else:
        with open(cfg['train_json_path'], 'r') as f:
            train_json = json.load(f)
    with open(cfg['test_json_path'], 'r') as f:
        test_json = json.load(f)
    max_eval_videos = cfg.get('max_eval_videos')
    if max_eval_videos is not None:
        test_json = test_json[:max_eval_videos]
    
    # Convert to per-window samples.
    train_windows = [] if evaluation_only else parse_json_to_windows(
        train_json,
        cfg['dataset_base_path'],
        cfg.get('window_length', 1.0),
        cfg.get('frames_per_window', 4),
        cfg.get('mode', 'binary')
    )
    
    test_windows = parse_json_to_windows(
        test_json,
        cfg['dataset_base_path'],
        cfg.get('window_length', 1.0),
        cfg.get('frames_per_window', 4),
        cfg.get('mode', 'binary'),
        str(Path(cfg['cache_dir']) / 'eval_windows') if cfg.get('cache_dir') else None,
        'Loading evaluation windows'
    )
    test_fake_segments = get_video_fake_segments(test_json)
    
    # Build datasets.
    train_dataset = None if evaluation_only else VideoWindowDatasetTrain(
        train_windows,
        mode=cfg.get('mode', 'binary'),
        transform_config=cfg.get('transform_config', {})
    )
    
    test_dataset = VideoWindowDatasetTest(
        test_windows,
        mode=cfg.get('mode', 'binary'),
        transform_config=cfg.get('transform_config', {}),
        task=cfg.get('task', 'normal')
    )
    
    # Build dataloaders.
    train_loader = None if evaluation_only else DataLoader(
        train_dataset,
        batch_size=cfg['train_batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    eval_start_index = int(cfg.get('eval_resume_start_index', 0))
    if eval_start_index < 0 or eval_start_index > len(test_dataset):
        raise ValueError(f"Invalid eval_resume_start_index={eval_start_index} for {len(test_dataset)} windows")
    eval_dataset = data.Subset(test_dataset, range(eval_start_index, len(test_dataset))) if eval_start_index else test_dataset
    test_loader = DataLoader(
        eval_dataset,
        batch_size=cfg['val_batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=False
    )


    
    if evaluation_only:
        print(f"******* Evaluation Videos: {len(test_json)}; Evaluation Windows: {len(test_windows)} *******")
    else:
        print(f"******* Training Windows: {len(train_windows)} -> Balanced: {len(train_dataset)}")
    
    return train_loader, test_loader, test_fake_segments
