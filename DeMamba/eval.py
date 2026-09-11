import os
import json
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import hashlib
import shutil
from pathlib import Path
from dataloader import generate_dataset_loader_from_json

import models
from util import build_model
from util import print_evaluation_metrics, eval_model

def parse_device_ids(value):
    """Parse explicit CUDA device IDs, e.g. ``0`` or ``0,1,...,7``."""
    try:
        device_ids = [int(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError('--device-ids must be comma-separated integers') from error
    if not device_ids or len(set(device_ids)) != len(device_ids) or min(device_ids) < 0:
        raise argparse.ArgumentTypeError('--device-ids must contain unique non-negative integers')
    return device_ids


def load_trained_model(cfg, model_path, device='cuda', device_ids=None):
    """Load a trained checkpoint."""
    model = build_model(
        cfg['model'],
        neuron_indices_path=cfg.get('neuron_indices_path'),
        xclip_model_path=cfg.get('xclip_model_path'),
        dinov3_repo_path=cfg.get('dinov3_repo_path'),
        dinov3_weights_path=cfg.get('dinov3_weights_path'),
        dinov3_model_name=cfg.get('dinov3_model_name'),
        dinov3_backend=cfg.get('dinov3_backend'),
        dinov3_hf_model_path=cfg.get('dinov3_hf_model_path'),
        dinov2_hf_model_path=cfg.get('dinov2_hf_model_path'),
    )
    model = model.to(device)
    
    if device_ids is not None and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Adapt between single-GPU and multi-GPU formats.
        if not any(key.startswith('module.') for key in state_dict.keys()) and device_ids is not None and len(device_ids) > 1:
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif any(key.startswith('module.') for key in state_dict.keys()) and (device_ids is None or len(device_ids) == 1):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"[ok] Loaded model: {model_path}")
    return model

def save_predictions_json(predictions_data, output_path):
    """Save predictions to a JSON file."""
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    serializable_data = convert_to_serializable(predictions_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"[ok] Predictions saved to: {output_path}")

def convert_to_serializable(obj):
    """Recursively convert NumPy values to JSON-compatible Python values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(value) for value in obj]
    return obj
def evaluate_trained_model(cfg, model_path, output_dir, device='cuda', device_ids=None):
    """Evaluate a trained model and persist the results."""
    
    # Create the output directory.
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model.
    model = load_trained_model(cfg, model_path, device, device_ids)
    
    # Build the dataloaders.
    print("******* Building dataloaders *******")
    _, val_loader, test_fake_segments = generate_dataset_loader_from_json(
        cfg, evaluation_only=bool(cfg.get('evaluation_only', False))
    )
    
    # Detect binary mode.
    is_binary = cfg.get('mode', '') == 'binary'
    
    # Loss used by `eval_model`.
    if cfg['mode'] == 'binary':
        loss_ce = torch.nn.BCEWithLogitsLoss()
    else:
        loss_ce = torch.nn.CrossEntropyLoss()
    
    # Run evaluation.
    print("******* Running evaluation *******")
    pred_accuracy, video_list, pred_labels, true_labels, outpred, all_videos_results = eval_model(
        cfg, model, val_loader, loss_ce, cfg.get('val_batch_size', 16), test_fake_segments
    )
    
    # Read the original test JSON.
    with open(cfg['test_json_path'], 'r', encoding='utf-8') as f:
        test_json_data = json.load(f)
    if cfg.get('max_eval_videos') is not None:
        test_json_data = test_json_data[:cfg['max_eval_videos']]
        
    final_results = []
    
    for video_item in test_json_data:
        # Copy the original record.
        new_item = video_item.copy()
        
        # Build the matching key.
        video_path = video_item["video_path"]
        formatted_video_name = video_path[:-4].replace('/', '_')
        
        # Look up the prediction.
        if formatted_video_name in all_videos_results['per_video_results']:
            video_results = all_videos_results['per_video_results'][formatted_video_name]
            pred_segments = video_results.get('pred_segments', [])
            
            # Build `model_inference`.
            model_inference = {}
            if pred_segments:
                model_inference['type'] = 'fake'
                # Make sure each segment is a list (not a tuple).
                model_inference['segment'] = [list(seg) for seg in pred_segments]
            else:
                model_inference['type'] = 'real'
                model_inference['segment'] = []
                
            new_item['model_inference'] = model_inference
        else:
            # If no prediction was produced, default to real.
            new_item['model_inference'] = {
                'type': 'real',
                'segment': []
            }
            
        final_results.append(new_item)
    
    # Save the new predictions JSON.
    predictions_json_path = os.path.join(output_dir, 'predictions.json')
    save_predictions_json(final_results, predictions_json_path)
    
    predictions_data = final_results
    
    # Optional: keep saving the legacy detailed results too.
    results_to_save = {
        'config': cfg,
        'model_path': model_path,
        'evaluation_results': {
            'overall_metrics': all_videos_results['overall_metrics'],
            'per_video_results': {}
        }
    }
    
    # Make per-video results JSON-serializable.
    for video_name, video_results in all_videos_results['per_video_results'].items():
        # Cast numpy scalars/arrays to native Python types.
        serializable_results = {}
        for key, value in video_results.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                serializable_results[key] = int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                serializable_results[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, list):
                # Convert tuple segments to lists.
                if key == 'pred_segments' and value and isinstance(value[0], tuple):
                    serializable_results[key] = [list(seg) for seg in value]
                else:
                    serializable_results[key] = value
            else:
                serializable_results[key] = value
        results_to_save['evaluation_results']['per_video_results'][video_name] = serializable_results
    
    # Save detailed results JSON.
    output_json_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(results_to_save), f, indent=2, ensure_ascii=False)
    
    # Save the summary metrics.
    summary_results = {
        'model_path': model_path,
        'evaluation_time': pd.Timestamp.now().isoformat(),
        'overall_metrics': all_videos_results['overall_metrics']
    }
    
    summary_json_path = os.path.join(output_dir, 'summary_results.json')
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(summary_results), f, indent=2, ensure_ascii=False)
    
    # Save the per-video CSV.
    csv_data = []
    for video_name, video_results in all_videos_results['per_video_results'].items():
        row = {
            'video_name': video_name,
            'video_true_type': video_results['video_true_type'],
            'video_pred_type': video_results['video_pred_type'],
            'video_correct': video_results['video_correct'],
            'F1Det': video_results.get('F1Det', 0),
            'F1Det_precision': video_results.get('F1Det_precision', 0),
            'F1Det_recall': video_results.get('F1Det_recall', 0),
            'F1Loc': video_results.get('F1Loc', 0),
            'F1Loc@0.1': video_results.get('F1Loc@0.1', 0),
            'F1Loc@0.3': video_results.get('F1Loc@0.3', 0),
            'F1Loc@0.5': video_results.get('F1Loc@0.5', 0),
            'F1Loc@0.7': video_results.get('F1Loc@0.7', 0),
            'frame_f1': video_results.get('f1', 0),
            'frame_precision': video_results.get('precision', 0),
            'frame_recall': video_results.get('recall', 0),
            'loc_f1_0.1': video_results.get('loc_f1_0.1', 0),
            'loc_f1_0.3': video_results.get('loc_f1_0.3', 0),
            'loc_f1_0.5': video_results.get('loc_f1_0.5', 0),
            'loc_f1_0.7': video_results.get('loc_f1_0.7', 0),
            'avg_loc_f1': video_results.get('avg_loc_f1', 0)
        }
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, 'detailed_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"[ok] Evaluation done. Results saved to: {output_dir}")
    print(f"   - predictions JSON: {predictions_json_path}")
    print(f"   - detailed JSON:    {output_json_path}")
    print(f"   - summary JSON:     {summary_json_path}")
    print(f"   - per-video CSV:    {csv_path}")
    
    # Print the overall metrics.
    print("\n" + "=" * 60)
    print("Evaluation summary")
    print("=" * 60)
    print_evaluation_metrics(all_videos_results['overall_metrics'])
    
    return results_to_save, predictions_data


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Evaluate a trained temporal-segmentation model.')
    parser.add_argument('--config', required=True, help='Path to the config file.')
    parser.add_argument('--model_path', required=True, help='Path to the trained checkpoint.')
    parser.add_argument('--output_dir', required=True, help='Directory to write evaluation results to.')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda only for the current evaluator).')
    parser.add_argument('--device-ids', default='0', help='CUDA device IDs: 0 for one GPU, or 0,1,2,3,4,5,6,7 for eight GPUs')
    parser.add_argument('--val-batch-size', type=int, default=None, help='Optional override for cfg.val_batch_size')
    parser.add_argument('--anno-file', default=None,
                        help='Override cfg.test_json_path, e.g. run proposal inference on the training split')
    parser.add_argument('--max-eval-videos', type=int, default=None,
                        help='Use only the first N videos and write only those videos (debug only)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Optional DataLoader worker override; use 0 when debugging')
    parser.add_argument('--resume', action='store_true', help='Resume an interrupted evaluation from its progress file')
    parser.add_argument('--clean', action='store_true', help='Remove this output directory\'s known evaluation products and caches first')
    parser.add_argument('--save-progress', action='store_true', help='Save batch-level evaluation progress for later --resume')
    parser.add_argument('--cache-data', action='store_true', help='Cache constructed video windows for reuse')
    parser.add_argument('--dataset-base-path', default=None, help='Optional override for cfg.dataset_base_path')
    
    args = parser.parse_args()
    args.device_ids = parse_device_ids(args.device_ids)
    if args.device != 'cuda':
        raise ValueError('The current evaluation loop uses CUDA tensors; pass --device cuda.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the current DeMamba evaluation loop')
    available = torch.cuda.device_count()
    if max(args.device_ids) >= available:
        raise ValueError(f'Requested --device-ids {args.device_ids}, but only {available} CUDA device(s) are visible')
    primary_device = args.device_ids[0]
    torch.cuda.set_device(primary_device)
    args.device = f'cuda:{primary_device}'
    
    # Sanity-check input paths.
    assert os.path.exists(args.config), f"Config file not found: {args.config}"
    assert os.path.exists(args.model_path), f"Checkpoint file not found: {args.model_path}"
    
    # Load the config.
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    if args.val_batch_size is not None:
        if args.val_batch_size <= 0:
            raise ValueError('--val-batch-size must be positive')
        cfg['val_batch_size'] = args.val_batch_size
    if args.anno_file is not None:
        if not os.path.isfile(args.anno_file):
            raise FileNotFoundError(f'Annotation file not found: {args.anno_file}')
        cfg['test_json_path'] = args.anno_file
    if args.max_eval_videos is not None:
        if args.max_eval_videos <= 0:
            raise ValueError('--max-eval-videos must be positive')
        cfg['max_eval_videos'] = args.max_eval_videos
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError('--num-workers must be non-negative')
        cfg['num_workers'] = args.num_workers
    if args.dataset_base_path is not None:
        cfg['dataset_base_path'] = args.dataset_base_path
    cfg['evaluation_only'] = bool(args.anno_file or args.max_eval_videos is not None or args.resume or args.save_progress)

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.clean and args.resume:
        raise ValueError('--clean and --resume cannot be used together')
    known_outputs = (
        'predictions.json', 'evaluation_results.json', 'summary_results.json',
        'detailed_results.csv', 'eval_progress.pt', 'eval_progress.pt.tmp',
    )
    if args.clean:
        for name in known_outputs:
            path = output_root / name
            if path.is_file():
                path.unlink()
        cache_root = output_root / 'data_cache'
        if cache_root.is_dir():
            shutil.rmtree(cache_root)
    if args.cache_data or args.resume:
        cfg['cache_dir'] = str(output_root / 'data_cache')
    signature_payload = {
        'annotation': str(Path(cfg['test_json_path']).resolve()),
        'annotation_mtime_ns': Path(cfg['test_json_path']).stat().st_mtime_ns,
        'model': str(Path(args.model_path).resolve()),
        'model_mtime_ns': Path(args.model_path).stat().st_mtime_ns,
        'max_eval_videos': cfg.get('max_eval_videos'),
        'val_batch_size': cfg.get('val_batch_size'),
    }
    cfg['eval_signature'] = hashlib.sha1(json.dumps(signature_payload, sort_keys=True).encode('utf-8')).hexdigest()
    cfg['eval_progress_path'] = str(output_root / 'eval_progress.pt') if (args.save_progress or args.resume) else None
    cfg['resume_eval'] = args.resume
    if args.resume and Path(cfg['eval_progress_path']).is_file():
        progress = torch.load(cfg['eval_progress_path'], map_location='cpu', weights_only=False)
        if progress.get('signature') != cfg['eval_signature']:
            raise ValueError('Evaluation progress belongs to different inputs; use --clean')
        cfg['eval_resume_start_index'] = int(progress.get('next_sample_index', 0))
        print(f"[resume] Continuing evaluation at window {cfg['eval_resume_start_index']}")
    
    print("Temporal-segmentation model evaluation")
    print("=" * 50)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device:     {args.device}  IDs: {args.device_ids}")
    print("=" * 50)
    
    # Run evaluation.
    results, predictions_data = evaluate_trained_model(cfg, args.model_path, args.output_dir, args.device, args.device_ids)
    
    print("[ok] Evaluation pipeline finished.")


if __name__ == '__main__':
    main()
